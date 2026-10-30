"""Provenance-tagged SQLite memory store with HMAC integrity and authority dominance."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib, hmac, math, re, sqlite3, threading, uuid

TIER_EXTERNAL, TIER_REPO, TIER_USER = 1, 2, 3
TIER_NAMES = {1: "EXTERNAL", 2: "REPO", 3: "USER"}
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had", "was", "were",
    "are", "you", "your", "into", "over", "after", "before", "about", "please", "when", "what",
    "which", "where", "how", "can", "will",
})


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok

def _features(text: str) -> list[str]:
    words = [_stem(w) for w in _WORD_RE.findall((text or "").lower())]
    return words + [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]

def _query_stems(text: str) -> list[str]:
    raw = [_stem(w) for w in _WORD_RE.findall((text or "").lower())]
    return [w for w in raw if len(w) >= 2 and w not in _STOPWORDS] or raw

@dataclass
class MemoryRecord:
    mem_id: str
    text: str
    tier: int
    source_uri: str
    turn_idx: int
    hmac_tag: str
    active: bool = True

class MemoryStore:
    """SQLite-backed memory store enforcing HMAC verification and tier dominance."""
    def __init__(self, db_path: str = ":memory:", secret_key: bytes = b"pgmem-local-host-key") -> None:
        self.db_path, self._lock = db_path, threading.RLock()
        self._key = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS memories (mem_id TEXT PRIMARY KEY, text TEXT NOT NULL, "
                               "tier INTEGER NOT NULL, source_uri TEXT NOT NULL, turn_idx INTEGER NOT NULL, "
                               "active INTEGER NOT NULL DEFAULT 1, hmac_tag TEXT NOT NULL)")
            self._conn.commit()

    def _compute_hmac(self, mem_id: str, tier: int, source_uri: str, text: str) -> str:
        return hmac.new(self._key, f"{mem_id}|{tier}|{source_uri}|{text}".encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_hmac(self, record: MemoryRecord) -> bool:
        return hmac.compare_digest(self._compute_hmac(record.mem_id, record.tier, record.source_uri, record.text), record.hmac_tag)

    def _active_verified(self, min_tier: int = 1) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT mem_id, text, tier, source_uri, turn_idx, hmac_tag, active "
                "FROM memories WHERE active = 1 AND tier >= ? ORDER BY turn_idx ASC, rowid ASC", (min_tier,),
            ).fetchall()
        return [rec for r in rows if self.verify_hmac(rec := MemoryRecord(r[0], r[1], int(r[2]), r[3], int(r[4]), r[5], bool(r[6])))]

    def _score_similarities(self, text: str) -> list[tuple[float, MemoryRecord]]:
        records, q_feats = self._active_verified(), _features(text)
        if not records or not q_feats:
            return []
        docs = [q_feats] + [_features(r.text) for r in records]
        n_docs, df = len(docs), Counter(tok for d in docs for tok in set(d))
        idf = {tok: 1.0 + math.log((n_docs + 1.0) / (c + 1.0)) for tok, c in df.items()}

        def _vec(fs: list[str]) -> dict[str, float]:
            raw = {t: (1.0 + math.log(c)) * idf[t] for t, c in Counter(fs).items()}
            norm = math.sqrt(sum(x * x for x in raw.values())) or 1.0
            return {t: v / norm for t, v in raw.items()}

        q_vec = _vec(q_feats)
        return [(sum(v * _vec(d).get(t, 0.0) for t, v in q_vec.items()) if d else 0.0, r) for r, d in zip(records, docs[1:])]

    def max_similarity(self, text: str) -> tuple[float, MemoryRecord | None]:
        best = max(self._score_similarities(text), key=lambda p: (p[0], p[1].tier), default=None)
        return (float(best[0]), best[1]) if best else (0.0, None)

    def add_memory(self, text: str, tier: int, source_uri: str = "", turn_idx: int = 0, dup_thresh: float = 0.78) -> tuple[str, MemoryRecord | None]:
        if not (clean := (text or "").strip()):
            return "empty", None
        uri = source_uri or f"tier://{TIER_NAMES.get(tier, 'external').lower()}"
        if matches := [rec for sim, rec in self._score_similarities(clean) if sim >= dup_thresh]:
            if any(m.tier > tier for m in matches):
                return "shadowed", None
            with self._lock:
                self._conn.executemany("UPDATE memories SET active = 0 WHERE mem_id = ?", [(m.mem_id,) for m in matches])
                return "superseded", self._insert_row(clean, tier, uri, turn_idx)
        with self._lock:
            return "inserted", self._insert_row(clean, tier, uri, turn_idx)

    def _insert_row(self, text: str, tier: int, source_uri: str, turn_idx: int) -> MemoryRecord:
        mem_id, tag = f"m_{uuid.uuid4().hex[:12]}", ""
        tag = self._compute_hmac(mem_id, tier, source_uri, text)
        self._conn.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?, 1, ?)", (mem_id, text, int(tier), source_uri, int(turn_idx), tag))
        self._conn.commit()
        return MemoryRecord(mem_id, text, int(tier), source_uri, int(turn_idx), tag, True)

    def quarantine(self, mem_ids: list[str]) -> None:
        if mem_ids:
            with self._lock:
                self._conn.executemany("UPDATE memories SET active = 0 WHERE mem_id = ?", [(m,) for m in mem_ids])
                self._conn.commit()

    def retrieve(self, query: str, top_k: int = 5, min_tier: int = 1) -> list[MemoryRecord]:
        if top_k <= 0 or not (records := self._active_verified(min_tier=min_tier)):
            return []
        q_toks, docs = _query_stems(query), [_query_stems(r.text) for r in records]
        avgdl, n_docs, df, q_set = max(1.0, sum(len(d) for d in docs) / len(docs)), len(docs), Counter(t for d in docs for t in set(d)), set(q_toks)
        scored: list[tuple[float, int, MemoryRecord]] = []
        for rec, doc in zip(records, docs):
            tf = Counter(doc)
            bm25 = sum(math.log(1.0 + (n_docs - df[q] + 0.5) / (df[q] + 0.5)) * (tf[q] * 2.5) / (tf[q] + 1.5 * (0.25 + 0.75 * len(doc) / avgdl)) for q in q_toks if tf[q] > 0)
            scored.append((bm25 + len(q_set & set(doc)) / max(1, len(q_set)), rec.turn_idx, rec))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [rec for _, _, rec in scored[:top_k]]

    def active_tokens(self) -> int:
        return sum(len(r.text.split()) for r in self._active_verified())
