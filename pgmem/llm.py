"""Thread-safe stdlib Gemini REST client with WAL SQLite prompt cache."""
from __future__ import annotations

import hashlib, json, os, re, sqlite3, threading, time, urllib.error, urllib.request
from typing import Any

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
_EXPORT_RE = re.compile(r"^\s*export\s+(?:GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.+?)\s*$")


def load_api_key() -> str:
    """Load Gemini API key from env or ~/.bashrc in memory without logging secrets."""
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key and os.path.isfile(bashrc := os.path.expanduser("~/.bashrc")):
        with open(bashrc, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if (m := _EXPORT_RE.match(line)) and (v := m.group(1).strip().strip("'\"")) and not v.startswith("$"):
                    key = v
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment or ~/.bashrc.")
    return key


def parse_json_payload(raw: str) -> Any:
    """Extract JSON object or list from model text, tolerating markdown fences."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.DOTALL).strip()
    for cand in [text] + [m.group(0) for pat in (r"\{.*\}", r"\[.*\]") if (m := re.search(pat, text, re.DOTALL))]:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    return {}


class GeminiClient:
    """Minimal Gemini REST client with SQLite WAL cache and rate-limit retries."""
    def __init__(self, model: str = "gemini-3.5-flash-lite", cache_path: str | None = None,
                 max_rps: float = 18.0, use_cache: bool = True, api_key: str | None = None) -> None:
        self.api_key = (api_key or load_api_key()).strip()
        self.model, self.max_rps, self.use_cache = model, max_rps, use_cache
        self._db_lock, self._rate_lock, self._next_ts = threading.Lock(), threading.Lock(), 0.0
        self.live_calls = self.cache_hits = self.total_prompt_tokens = self.total_output_tokens = 0
        self.cache_path = cache_path or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), ".cache", "llm_cache.sqlite")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def generate(self, prompt: str, *, system_instruction: str = "", json_mode: bool = False,
                 temperature: float = 0.0, max_output_tokens: int = 1024, model: str | None = None,
                 cache_salt: str = "", use_cache: bool | None = None) -> dict[str, Any]:
        target, allow = model or self.model, (self.use_cache if use_cache is None else use_cache)
        ckey = hashlib.sha256(json.dumps({"v": "pgmem_v1", "m": target, "s": system_instruction, "p": prompt,
                                          "j": json_mode, "t": temperature, "k": max_output_tokens, "x": cache_salt}, sort_keys=True).encode("utf-8")).hexdigest()
        if allow:
            with self._db_lock, sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                if row := conn.execute("SELECT value FROM cache WHERE key = ?", (ckey,)).fetchone():
                    self.cache_hits += 1
                    return json.loads(row[0])
        cfg: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        payload, last_err = json.dumps(body).encode("utf-8"), "unknown error"
        for delay in (0.0, 1.5, 4.0, 9.0, 18.0, 32.0):
            if delay > 0:
                time.sleep(delay)
            if self.max_rps > 0:
                with self._rate_lock:
                    now = time.monotonic()
                    wait, self._next_ts = self._next_ts - now, max(now, self._next_ts) + 1.0 / self.max_rps
                if wait > 0:
                    time.sleep(wait)
            req = urllib.request.Request(API_URL.format(model=target, key=self.api_key), data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=65) as resp:
                    raw_resp = json.loads(resp.read().decode("utf-8"))
                cands, usage = raw_resp.get("candidates") or [], raw_resp.get("usageMetadata") or {}
                parts = [p["text"] for c in cands for p in (c.get("content") or {}).get("parts") or [] if not p.get("thought") and "text" in p]
                out = {"text": "".join(parts).strip(), "prompt_tokens": int(usage.get("promptTokenCount") or 0), "output_tokens": int(usage.get("candidatesTokenCount") or 0)}
                with self._db_lock:
                    self.live_calls += 1
                    self.total_prompt_tokens += out["prompt_tokens"]
                    self.total_output_tokens += out["output_tokens"]
                    if allow and out["text"] and (str(cands[0].get("finishReason") or "") if cands else "") != "MAX_TOKENS":
                        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (ckey, json.dumps(out, sort_keys=True)))
                return out
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace").replace(self.api_key, "[REDACTED]")
                last_err = f"HTTP {e.code}: {detail[:600]}"
                if e.code not in (429, 500, 502, 503, 504) and not (e.code == 400 and "dependency" in detail.lower()):
                    raise RuntimeError(last_err) from None
            except Exception as e:
                last_err = str(e).replace(self.api_key, "[REDACTED]")
        raise RuntimeError(f"Gemini request failed after retries: {last_err}")
