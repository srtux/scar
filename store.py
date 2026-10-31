"""Origin-tagged SQLite memory store for MemoryGuard.

What this file does (in plain English):
1. Saves agent memory notes in a SQLite table (`memories`), where every note is
   labeled with a trust tier showing where it came from:
     - TIER_USER (3): Spoken or typed directly by the user (highest trust).
     - TIER_REPO (2): Checked-in repository docs like README.md or CONTRIBUTING.md.
     - TIER_EXTERNAL (1): Untrusted text from GitHub issues, PRs, or web pages.

2. Attaches a cryptographic "Origin Wristband" (`hmac_tag`) to every row using
   HMAC-SHA256. When reading memories back, we recompute the wristband tag and
   skip any row whose text or trust tier was modified in the database.

3. Enforces "Authority Dominance" when duplicates arrive:
     - If an untrusted EXTERNAL note copies an existing USER or REPO rule
       (similarity >= 0.78), we ignore the untrusted copy ("shadowed") so an
       external page can never taint or overwrite a user's own rule.
     - If an equal-trust or higher-trust note updates an older note, the old row
       is marked inactive (`active = 0`) and replaced ("superseded").

4. Retrieves the top matching notes for a new question using standard BM25
   keyword scoring (no external vector database needed).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
import math
import re
import sqlite3
import threading
import uuid

# Trust tiers ordered from lowest trust (1) to highest trust (3).
TIER_EXTERNAL = 1
TIER_REPO = 2
TIER_USER = 3

TIER_NAMES = {
    TIER_EXTERNAL: "EXTERNAL",
    TIER_REPO: "REPO",
    TIER_USER: "USER",
}

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common filler words ignored during keyword search so important nouns/flags win.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "was", "were", "are", "you", "your", "into", "over", "after", "before",
    "about", "please", "when", "what", "which", "where", "how", "can", "will",
})


def _stem(word: str) -> str:
    """Strip common English suffixes so 'testing' and 'tests' both match 'test'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _features(text: str) -> list[str]:
    """Turn text into stemmed words plus 2-word pairs (bigrams) for duplicate checks."""
    words = [_stem(w) for w in _WORD_RE.findall((text or "").lower())]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _query_stems(text: str) -> list[str]:
    """Extract meaningful stemmed search words from a query or memory note."""
    all_words = [_stem(w) for w in _WORD_RE.findall((text or "").lower())]
    filtered = [w for w in all_words if len(w) >= 2 and w not in _STOPWORDS]
    return filtered if filtered else all_words


@dataclass
class MemoryRecord:
    """One verified memory note stored in MemoryStore."""

    mem_id: str
    text: str
    tier: int
    source_uri: str
    turn_idx: int
    hmac_tag: str
    active: bool = True


class MemoryStore:
    """SQLite-backed memory ledger with HMAC tamper checks and trust-tier rules."""

    def __init__(
        self,
        db_path: str = ":memory:",
        secret_key: bytes = b"memory_guard-local-host-key",
    ) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        if isinstance(secret_key, str):
            self._key = secret_key.encode("utf-8")
        else:
            self._key = bytes(secret_key)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    mem_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    source_uri TEXT NOT NULL,
                    turn_idx INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    hmac_tag TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def _compute_hmac(self, mem_id: str, tier: int, source_uri: str, text: str) -> str:
        """Create a tamper-evident HMAC-SHA256 wristband signature over all key fields."""
        payload = f"{mem_id}|{tier}|{source_uri}|{text}".encode("utf-8")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify_hmac(self, record: MemoryRecord) -> bool:
        """Return True only if the record's wristband matches its text and trust tier."""
        expected_tag = self._compute_hmac(
            record.mem_id,
            record.tier,
            record.source_uri,
            record.text,
        )
        return hmac.compare_digest(expected_tag, record.hmac_tag)

    def _active_verified(self, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Load all active memories at or above `min_tier` whose HMAC wristband is valid."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT mem_id, text, tier, source_uri, turn_idx, hmac_tag, active
                FROM memories
                WHERE active = 1 AND tier >= ?
                ORDER BY turn_idx ASC, rowid ASC
                """,
                (min_tier,),
            ).fetchall()

        verified_records: list[MemoryRecord] = []
        for row in rows:
            record = MemoryRecord(
                mem_id=row[0],
                text=row[1],
                tier=int(row[2]),
                source_uri=row[3],
                turn_idx=int(row[4]),
                hmac_tag=row[5],
                active=bool(row[6]),
            )
            # Silently skip any row that was tampered with directly in SQLite.
            if self.verify_hmac(record):
                verified_records.append(record)
        return verified_records

    def _score_similarities(self, text: str) -> list[tuple[float, MemoryRecord]]:
        """Compute TF-IDF cosine similarity (0.0 to 1.0) between `text` and each active memory."""
        records = self._active_verified()
        query_features = _features(text)
        if not records or not query_features:
            return []

        record_feature_lists = [_features(rec.text) for rec in records]
        all_docs = [query_features] + record_feature_lists
        num_docs = len(all_docs)

        # Count how many documents contain each word/bigram so rare terms get higher weight.
        doc_freq: Counter[str] = Counter()
        for doc in all_docs:
            for token in set(doc):
                doc_freq[token] += 1

        inverse_doc_freq = {
            token: 1.0 + math.log((num_docs + 1.0) / (count + 1.0))
            for token, count in doc_freq.items()
        }

        def build_unit_vector(features: list[str]) -> dict[str, float]:
            raw_weights = {
                token: (1.0 + math.log(count)) * inverse_doc_freq[token]
                for token, count in Counter(features).items()
            }
            length = math.sqrt(sum(weight * weight for weight in raw_weights.values())) or 1.0
            return {token: weight / length for token, weight in raw_weights.items()}

        query_vec = build_unit_vector(query_features)
        scored: list[tuple[float, MemoryRecord]] = []
        for record, doc_features in zip(records, record_feature_lists):
            if not doc_features:
                scored.append((0.0, record))
                continue
            doc_vec = build_unit_vector(doc_features)
            cosine_sim = sum(
                q_weight * doc_vec.get(token, 0.0)
                for token, q_weight in query_vec.items()
            )
            scored.append((cosine_sim, record))
        return scored

    def max_similarity(self, text: str) -> tuple[float, MemoryRecord | None]:
        """Return the highest similarity score (and matching record) already in memory."""
        scored = self._score_similarities(text)
        if not scored:
            return 0.0, None
        best_sim, best_record = max(scored, key=lambda pair: (pair[0], pair[1].tier))
        return float(best_sim), best_record

    def add_memory(
        self,
        text: str,
        tier: int,
        source_uri: str = "",
        turn_idx: int = 0,
        dup_thresh: float = 0.78,
    ) -> tuple[str, MemoryRecord | None]:
        """Save a note while enforcing trust-tier duplicate rules.

        Returns `(status, record)` where `status` is one of:
          - "empty": blank text was ignored.
          - "shadowed": a higher-trust rule already says the same thing, so this
            lower-trust duplicate was dropped.
          - "superseded": older equal-or-lower-trust duplicates were deactivated
            and replaced by this newer note.
          - "inserted": saved as a brand-new memory note.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return "empty", None

        tier_label = TIER_NAMES.get(tier, "external").lower()
        resolved_uri = source_uri or f"tier://{tier_label}"

        near_duplicates = [
            record
            for similarity, record in self._score_similarities(clean_text)
            if similarity >= dup_thresh
        ]
        if near_duplicates:
            # Authority Dominance: never let a low-tier note overwrite a higher-tier rule.
            if any(existing.tier > tier for existing in near_duplicates):
                return "shadowed", None

            with self._lock:
                rows_to_retire = [(existing.mem_id,) for existing in near_duplicates]
                self._conn.executemany(
                    "UPDATE memories SET active = 0 WHERE mem_id = ?",
                    rows_to_retire,
                )
                new_record = self._insert_row(clean_text, tier, resolved_uri, turn_idx)
                return "superseded", new_record

        with self._lock:
            new_record = self._insert_row(clean_text, tier, resolved_uri, turn_idx)
            return "inserted", new_record

    def _insert_row(
        self,
        text: str,
        tier: int,
        source_uri: str,
        turn_idx: int,
    ) -> MemoryRecord:
        """Sign and insert one row into SQLite (caller holds `self._lock`)."""
        mem_id = f"m_{uuid.uuid4().hex[:12]}"
        hmac_tag = self._compute_hmac(mem_id, tier, source_uri, text)
        self._conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, 1, ?)",
            (mem_id, text, int(tier), source_uri, int(turn_idx), hmac_tag),
        )
        self._conn.commit()
        return MemoryRecord(
            mem_id=mem_id,
            text=text,
            tier=int(tier),
            source_uri=source_uri,
            turn_idx=int(turn_idx),
            hmac_tag=hmac_tag,
            active=True,
        )

    def quarantine(self, mem_ids: list[str]) -> None:
        """Deactivate specific memory IDs (used when Clean-Room catches a poisoned note)."""
        if not mem_ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET active = 0 WHERE mem_id = ?",
                [(mem_id,) for mem_id in mem_ids],
            )
            self._conn.commit()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_tier: int = TIER_EXTERNAL,
    ) -> list[MemoryRecord]:
        """Return up to `top_k` verified memories ranked by BM25 keyword relevance."""
        if top_k <= 0:
            return []
        records = self._active_verified(min_tier=min_tier)
        if not records:
            return []

        query_tokens = _query_stems(query)
        query_token_set = set(query_tokens)
        doc_tokens_list = [_query_stems(rec.text) for rec in records]

        num_docs = len(doc_tokens_list)
        avg_doc_len = max(1.0, sum(len(doc) for doc in doc_tokens_list) / num_docs)

        doc_freq: Counter[str] = Counter()
        for doc in doc_tokens_list:
            for token in set(doc):
                doc_freq[token] += 1

        # Standard BM25 parameters: k1 = 1.5 controls term saturation, b = 0.75 normalizes length.
        scored: list[tuple[float, int, MemoryRecord]] = []
        for record, doc_tokens in zip(records, doc_tokens_list):
            term_counts = Counter(doc_tokens)
            bm25_score = 0.0
            for q_tok in query_tokens:
                tf = term_counts[q_tok]
                if tf == 0:
                    continue
                idf = math.log(1.0 + (num_docs - doc_freq[q_tok] + 0.5) / (doc_freq[q_tok] + 0.5))
                length_norm = 0.25 + 0.75 * (len(doc_tokens) / avg_doc_len)
                bm25_score += idf * (tf * 2.5) / (tf + 1.5 * length_norm)

            overlap_fraction = len(query_token_set & set(doc_tokens)) / max(1, len(query_token_set))
            total_score = bm25_score + overlap_fraction
            scored.append((total_score, record.turn_idx, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:top_k]]

    def active_tokens(self) -> int:
        """Count total whitespace-separated words across all active, verified memories."""
        return sum(len(record.text.split()) for record in self._active_verified())
