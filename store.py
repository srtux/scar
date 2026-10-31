"""Origin-signed SQLite + FTS5 memory store for MemoryGuard (`memory_guard`).

What this file does (in plain English):
1. Stores each memory note in SQLite with a source trust tier:
     - TIER_USER (3): Typed or spoken directly by the user (highest trust).
     - TIER_REPO (2): Checked-in repository docs like README.md or CONTRIBUTING.md.
     - TIER_EXTERNAL (1): Untrusted text from GitHub issues, PRs, or web pages.

2. Signs every row (`mem_id`, `tier`, `source_uri`, `active`, `text`) with
   HMAC-SHA256 using a random 256-bit host secret (`secrets.token_bytes(32)`).
   Direct edits to SQLite (changing a note's tier, text, or active status)
   fail signature verification on read and increment `tampered_count()`.

3. Prevents untrusted clones and accidental rule overwrites:
     - Authority Dominance: if an untrusted EXTERNAL note copies an existing
       USER or REPO rule (`similarity >= 0.78`), it is dropped as `"shadowed"`.
     - Identifier & Polarity Guard: two notes that target different files/flags
       (like `tests/test_auth.py` vs `tests/test_billing.py`) or opposite
       polarity (`Always` vs `Never`) score `0.0` so distinct rules never
       accidentally overwrite one another.

4. Uses SQLite's built-in C `FTS5` full-text engine (`bm25()`) for fast indexed
   keyword retrieval with Porter stemming.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
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

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Matches file paths, dotted filenames/versions, module:attr targets, CLI flags, and numbers.
_IDENTIFIER_RE = re.compile(
    r"(?:https?://\S+"
    r"|\b[\w.-]+(?:/[\w./-]+|\.[a-zA-Z0-9]{1,8}|:[\w.-]+)"
    r"|(?<!\w)--?[a-zA-Z][\w-]*(?:=\S+)?"
    r"|\b\d+(?:\.\d+)*\b)"
)

_NEGATION_WORDS = frozenset({
    "never", "not", "no", "dont", "avoid", "disable", "without", "forbid",
})


def _extract_identifiers(text: str) -> set[str]:
    """Extract file paths, flags, URLs, and numbers that distinguish specific rules."""
    raw_matches = _IDENTIFIER_RE.findall((text or "").lower())
    return {match.strip().rstrip(".,;:)") for match in raw_matches if match.strip()}


def _has_negation(text: str) -> bool:
    """Return True if the note contains a negation word ('never', 'not', 'avoid', etc.)."""
    words = set(_TOKEN_RE.findall((text or "").lower()))
    return bool(words & _NEGATION_WORDS)


def note_similarity(text_a: str, text_b: str) -> float:
    """Compare two notes (0.0 to 1.0) while respecting paths, flags, and negations."""
    norm_a = " ".join((text_a or "").lower().split())
    norm_b = " ".join((text_b or "").lower().split())
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    # Opposite instructions ("Always run X" vs "Never run X") are not duplicates.
    if _has_negation(norm_a) != _has_negation(norm_b):
        return 0.0

    # Rules pointing to different files, flags, or ports (e.g. test_auth.py vs test_billing.py)
    # must stay separate rather than overwriting each other.
    ids_a = _extract_identifiers(norm_a)
    ids_b = _extract_identifiers(norm_b)
    if (ids_a - ids_b) and (ids_b - ids_a):
        return 0.0

    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _resolve_secret_key(db_path: str, secret_key: bytes | str | None) -> bytes:
    """Resolve or generate a 256-bit HMAC key (never hardcoded in source code)."""
    if secret_key is not None:
        return secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)

    from llm import load_dotenv_file

    load_dotenv_file()
    env_key = os.environ.get("MEMORY_GUARD_HMAC_KEY", "").strip()
    if env_key:
        return env_key.encode("utf-8")

    if db_path == ":memory:":
        return secrets.token_bytes(32)

    # For persistent SQLite files, keep an owner-only (0600) random key alongside the DB.
    key_path = Path(f"{db_path}.hmac_key")
    if key_path.is_file():
        return key_path.read_bytes()

    key_bytes = secrets.token_bytes(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key_bytes)
    return key_bytes


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
    """SQLite + FTS5 memory store with full-row HMAC verification and trust-tier rules."""

    def __init__(
        self,
        db_path: str = ":memory:",
        secret_key: bytes | str | None = None,
    ) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._key = _resolve_secret_key(db_path, secret_key)
        self._tampered_ids: set[str] = set()

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
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    mem_id UNINDEXED,
                    text,
                    tokenize='porter unicode61'
                )
                """
            )
            self._conn.commit()

    def _compute_hmac(
        self,
        mem_id: str,
        tier: int,
        source_uri: str,
        text: str,
        active: bool = True,
    ) -> str:
        """Sign every security-relevant column, including `active` status."""
        payload = f"{mem_id}|{int(tier)}|{source_uri}|{int(bool(active))}|{text}".encode("utf-8")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify_hmac(self, record: MemoryRecord) -> bool:
        """Return True only if text, tier, source_uri, and active flag match `hmac_tag`."""
        expected_tag = self._compute_hmac(
            record.mem_id,
            record.tier,
            record.source_uri,
            record.text,
            record.active,
        )
        return hmac.compare_digest(expected_tag, record.hmac_tag)

    def _active_verified(self, min_tier: int = TIER_EXTERNAL) -> list[MemoryRecord]:
        """Audit all rows for HMAC tampering and return verified active notes >= `min_tier`."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT mem_id, text, tier, source_uri, turn_idx, hmac_tag, active
                FROM memories
                ORDER BY turn_idx ASC, rowid ASC
                """
            ).fetchall()

        verified: list[MemoryRecord] = []
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
            if not self.verify_hmac(record):
                self._tampered_ids.add(record.mem_id)
                continue
            if record.active and record.tier >= min_tier:
                verified.append(record)
        return verified

    def tampered_count(self) -> int:
        """Return the number of tampered SQLite rows detected so far."""
        self._active_verified()
        return len(self._tampered_ids)

    def _score_similarities(self, text: str) -> list[tuple[float, MemoryRecord]]:
        """Score similarity (0.0 to 1.0) between `text` and each verified active memory."""
        records = self._active_verified()
        return [(note_similarity(text, rec.text), rec) for rec in records]

    def max_similarity(self, text: str) -> tuple[float, MemoryRecord | None]:
        """Return the highest similarity score (and matching record) in active memory."""
        scored = self._score_similarities(text)
        if not scored:
            return 0.0, None
        best_sim, best_record = max(scored, key=lambda pair: (pair[0], pair[1].tier))
        return float(best_sim), (best_record if best_sim > 0.0 else None)

    def _deactivate_verified_records(self, records: list[MemoryRecord]) -> None:
        """Mark verified records inactive (`active = 0`) with a freshly signed HMAC."""
        for rec in records:
            retired_hmac = self._compute_hmac(
                rec.mem_id, rec.tier, rec.source_uri, rec.text, active=False
            )
            self._conn.execute(
                "UPDATE memories SET active = 0, hmac_tag = ? WHERE mem_id = ?",
                (retired_hmac, rec.mem_id),
            )
            self._conn.execute(
                "DELETE FROM memories_fts WHERE mem_id = ?",
                (rec.mem_id,),
            )

    def add_memory(
        self,
        text: str,
        tier: int,
        source_uri: str = "",
        turn_idx: int = 0,
        dup_thresh: float = 0.78,
    ) -> tuple[str, MemoryRecord | None]:
        """Save a note while enforcing trust-tier duplicate rules."""
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
            # Authority Dominance: never let a lower-tier note overwrite a higher-tier rule.
            if any(existing.tier > tier for existing in near_duplicates):
                return "shadowed", None

            with self._lock:
                self._deactivate_verified_records(near_duplicates)
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
        """Sign and insert one active row into SQLite and the FTS5 BM25 index."""
        mem_id = f"m_{uuid.uuid4().hex[:12]}"
        hmac_tag = self._compute_hmac(mem_id, tier, source_uri, text, active=True)
        self._conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, 1, ?)",
            (mem_id, text, int(tier), source_uri, int(turn_idx), hmac_tag),
        )
        self._conn.execute(
            "INSERT INTO memories_fts (mem_id, text) VALUES (?, ?)",
            (mem_id, text),
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
        """Deactivate and re-sign specific memory IDs caught by Clean-Room."""
        if not mem_ids:
            return
        target_set = set(mem_ids)
        verified = [rec for rec in self._active_verified() if rec.mem_id in target_set]
        if not verified:
            return
        with self._lock:
            self._deactivate_verified_records(verified)
            self._conn.commit()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_tier: int = TIER_EXTERNAL,
    ) -> list[MemoryRecord]:
        """Retrieve up to `top_k` verified memories ranked by SQLite FTS5 BM25."""
        if top_k <= 0:
            return []
        verified_by_id = {
            rec.mem_id: rec for rec in self._active_verified(min_tier=min_tier)
        }
        if not verified_by_id:
            return []

        raw_tokens = [tok for tok in _TOKEN_RE.findall((query or "").lower()) if len(tok) >= 2]
        unique_tokens = list(dict.fromkeys(raw_tokens))

        ranked: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        if unique_tokens:
            # Double-quoting alphanumeric tokens makes FTS5 treat words like "not" or "or"
            # strictly as literal search terms rather than boolean syntax errors.
            fts_query = " OR ".join(f'"{tok}"' for tok in unique_tokens)
            with self._lock:
                fts_rows = self._conn.execute(
                    """
                    SELECT mem_id
                    FROM memories_fts
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25(memories_fts) ASC
                    """,
                    (fts_query,),
                ).fetchall()
            for (mem_id,) in fts_rows:
                rec = verified_by_id.get(mem_id)
                if rec is not None and mem_id not in seen_ids:
                    ranked.append(rec)
                    seen_ids.add(mem_id)
                    if len(ranked) >= top_k:
                        return ranked

        # If fewer than `top_k` notes matched keywords, include recent verified notes.
        remaining = sorted(
            (rec for mid, rec in verified_by_id.items() if mid not in seen_ids),
            key=lambda rec: rec.turn_idx,
            reverse=True,
        )
        ranked.extend(remaining[: top_k - len(ranked)])
        return ranked

    def active_tokens(self) -> int:
        """Count total whitespace-separated words across all verified active memories."""
        return sum(len(record.text.split()) for record in self._active_verified())
