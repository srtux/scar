"""Standard HTTPS LLM and Embedding client for SCAR (stdlib-only, OpenAI & Gemini REST).

Supports:
- Public Google Gemini Developer REST API via `GEMINI_API_KEY`
  (`gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-embedding-001`)
- OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` via `OPENAI_API_KEY`
  (and optional `OPENAI_BASE_URL`)
- Thread-safe local SQLite cache (`.cache/llm_cache.sqlite`) plus live call/token counters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from LLM text output."""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


class LLMClient:
    """Thread-safe LLM and embedding client with deterministic caching and usage accounting."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        self.openai_base_url = (
            openai_base_url
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        )
        if not self.gemini_api_key and not self.openai_api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY or OPENAI_API_KEY in environment to run SCAR LLM experiments."
            )

        self.backend = "gemini" if self.gemini_api_key else "openai"
        self.default_model = default_model or os.environ.get("SCAR_LLM_MODEL", "gemini-3.8-flash")
        self.fast_model = fast_model or os.environ.get("SCAR_FAST_MODEL", "gemini-3.5-flash-lite")
        self.embedding_model = embedding_model or os.environ.get(
            "SCAR_EMBED_MODEL",
            "gemini-embedding-001" if self.backend == "gemini" else "text-embedding-3-small",
        )

        self.use_cache = use_cache
        self.max_rps = float(os.environ.get("SCAR_MAX_RPS", "14.0"))
        self._rate_lock = threading.Lock()
        self._next_allowed_ts = 0.0
        self._lock = threading.Lock()
        self.stats = {
            "llm_api_calls": 0,
            "llm_cache_hits": 0,
            "embed_api_calls": 0,
            "embed_cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        if cache_path is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cache_dir = os.path.join(repo_root, ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "llm_cache.sqlite")
        self.cache_path = cache_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.commit()

    def _cache_get(self, key: str) -> Optional[Any]:
        if not self.use_cache:
            return None
        with self._lock:
            with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _cache_set(self, key: str, value: Any) -> None:
        if not self.use_cache:
            return
        payload = json.dumps(value, sort_keys=True)
        with self._lock:
            with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                    (key, payload),
                )
                conn.commit()

    def _throttle(self) -> None:
        if self.max_rps <= 0:
            return
        min_interval = 1.0 / self.max_rps
        with self._rate_lock:
            now = time.monotonic()
            wait_s = self._next_allowed_ts - now
            self._next_allowed_ts = max(now, self._next_allowed_ts) + min_interval
        if wait_s > 0:
            time.sleep(wait_s)

    def _post_json(self, url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        encoded = json.dumps(body).encode("utf-8")
        last_err: Optional[Exception] = None
        backoff_schedule = [2.0, 5.0, 10.0, 18.0, 30.0, 45.0, 62.0]
        for attempt in range(len(backoff_schedule) + 1):
            self._throttle()
            req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {e.code}: {err_body[:400]}")
                if e.code in (429, 500, 502, 503, 504) and attempt < len(backoff_schedule):
                    time.sleep(backoff_schedule[attempt])
                    continue
                raise last_err from e
            except Exception as e:
                last_err = e
                if attempt < len(backoff_schedule):
                    time.sleep(backoff_schedule[attempt])
                    continue
        raise RuntimeError(f"Request failed after retries: {last_err}")

    def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> str:
        """Generate single-turn text or structured JSON string."""
        use_model = model or self.default_model
        key_payload = {
            "op": "generate_text_v1",
            "backend": self.backend,
            "model": use_model,
            "sys": system_instruction,
            "prompt": prompt,
            "json_mode": json_mode,
            "temp": temperature,
            "max_tok": max_output_tokens,
        }
        cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        cached = self._cache_get(cache_key)
        if cached is not None:
            with self._lock:
                self.stats["llm_cache_hits"] += 1
            return str(cached.get("text", ""))

        finish_reason = ""
        if self.backend == "gemini":
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{use_model}:generateContent?key={self.gemini_api_key}"
            )
            gen_cfg: Dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": max(max_output_tokens, 4096),
            }
            if "gemini-3" in use_model or "gemini-4" in use_model:
                gen_cfg["thinkingConfig"] = {"thinkingLevel": "LOW"}
            if json_mode:
                gen_cfg["responseMimeType"] = "application/json"
            body: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            }
            if system_instruction:
                body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            resp = self._post_json(url, body, {"Content-Type": "application/json"})
            cands = resp.get("candidates") or []
            if cands:
                finish_reason = str(cands[0].get("finishReason") or "")
            text_parts: List[str] = []
            for cand in cands:
                for part in (cand.get("content") or {}).get("parts") or []:
                    if not part.get("thought") and "text" in part:
                        text_parts.append(part["text"])
            out_text = "".join(text_parts).strip()
            usage = resp.get("usageMetadata") or {}
            p_tok = int(usage.get("promptTokenCount") or 0)
            c_tok = int(usage.get("candidatesTokenCount") or 0)
        else:
            url = f"{self.openai_base_url}/chat/completions"
            msgs: List[Dict[str, str]] = []
            if system_instruction:
                msgs.append({"role": "system", "content": system_instruction})
            msgs.append({"role": "user", "content": prompt})
            body = {
                "model": use_model,
                "messages": msgs,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            resp = self._post_json(
                url,
                body,
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}",
                },
            )
            choice0 = (resp.get("choices") or [{}])[0]
            finish_reason = str(choice0.get("finish_reason") or "")
            out_text = (choice0.get("message") or {}).get("content") or ""
            usage = resp.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens") or 0)
            c_tok = int(usage.get("completion_tokens") or 0)

        with self._lock:
            self.stats["llm_api_calls"] += 1
            self.stats["prompt_tokens"] += p_tok
            self.stats["completion_tokens"] += c_tok
        if out_text and finish_reason not in ("MAX_TOKENS", "length"):
            self._cache_set(cache_key, {"text": out_text, "p_tok": p_tok, "c_tok": c_tok})
        return out_text

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        model: Optional[str] = None,
        max_output_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON dictionary."""
        raw = self.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model,
            json_mode=True,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )
        return parse_json_object(raw)

    @staticmethod
    def _tau2_messages_to_gemini_contents(
        messages: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert multi-turn assistant/user/tool history into alternating Gemini contents.

        Serializes prior foreign-model tool calls and JSON tool results as structured text
        blocks so Gemini 3.x never rejects foreign histories for lacking Gemini-internal
        `thought_signature` tokens, while preserving native `functionDeclarations` tool
        calling on the live evaluated turn.
        """
        call_id_to_name: Dict[str, str] = {}
        raw_turns: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            tcs = m.get("tool_calls") or []
            if role == "assistant":
                blocks: List[str] = []
                if content:
                    blocks.append(content)
                for tc in tcs:
                    tc_id = str(tc.get("id") or "")
                    tc_name = str(tc.get("name") or "unknown_tool")
                    tc_args = tc.get("arguments")
                    if not isinstance(tc_args, str):
                        tc_args_str = json.dumps(tc_args or {}, sort_keys=True)
                    else:
                        tc_args_str = tc_args
                    if tc_id:
                        call_id_to_name[tc_id] = tc_name
                    blocks.append(f"[Prior Tool Call: {tc_name}({tc_args_str})]")
                if blocks:
                    raw_turns.append({"role": "model", "parts": [{"text": "\n".join(blocks)}]})
            elif role == "user":
                if content:
                    raw_turns.append({"role": "user", "parts": [{"text": content}]})
            elif role == "tool":
                tc_id = str(m.get("id") or m.get("tool_call_id") or "")
                tc_name = m.get("tool_name") or call_id_to_name.get(tc_id, "tool_result")
                err_tag = " [ERROR]" if m.get("error") else ""
                raw_turns.append(
                    {
                        "role": "user",
                        "parts": [{"text": f"[Tool Output ({tc_name}){err_tag}]: {content}"}],
                    }
                )

        merged: List[Dict[str, Any]] = []
        for turn in raw_turns:
            if merged and merged[-1]["role"] == turn["role"]:
                merged[-1]["parts"].extend(turn["parts"])
            else:
                merged.append({"role": turn["role"], "parts": list(turn["parts"])})

        if merged and merged[0]["role"] == "model":
            merged.insert(0, {"role": "user", "parts": [{"text": "[Customer connected to support chat]"}]})
        if not merged:
            merged.append({"role": "user", "parts": [{"text": "Hello"}]})
        return merged

    def generate_agent_step(
        self,
        messages: Sequence[Dict[str, Any]],
        system_instruction: str,
        tools: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Run one live agent decision turn with native tool calling."""
        use_model = model or self.default_model
        key_payload = {
            "op": "agent_step_v1",
            "backend": self.backend,
            "model": use_model,
            "sys": system_instruction,
            "messages": messages,
            "tools": tools,
            "temp": temperature,
        }
        cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        cached = self._cache_get(cache_key)
        if cached is not None:
            with self._lock:
                self.stats["llm_cache_hits"] += 1
            return dict(cached)

        finish_reason = ""
        if self.backend == "gemini":
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{use_model}:generateContent?key={self.gemini_api_key}"
            )
            contents = self._tau2_messages_to_gemini_contents(messages)
            step_gen_cfg: Dict[str, Any] = {
                "temperature": temperature,
                "maxOutputTokens": 4096,
            }
            if "gemini-3" in use_model or "gemini-4" in use_model:
                step_gen_cfg["thinkingConfig"] = {"thinkingLevel": "LOW"}
            body: Dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": contents,
                "tools": [{"functionDeclarations": list(tools)}],
                "generationConfig": step_gen_cfg,
            }
            resp = self._post_json(url, body, {"Content-Type": "application/json"})
            cands = resp.get("candidates") or []
            if cands:
                finish_reason = str(cands[0].get("finishReason") or "")
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for cand in cands:
                for part in (cand.get("content") or {}).get("parts") or []:
                    if part.get("thought"):
                        continue
                    if "text" in part:
                        text_parts.append(part["text"])
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append(
                            {
                                "name": fc.get("name", ""),
                                "arguments": fc.get("args") or {},
                            }
                        )
            usage = resp.get("usageMetadata") or {}
            p_tok = int(usage.get("promptTokenCount") or 0)
            c_tok = int(usage.get("candidatesTokenCount") or 0)
            result = {"content": "".join(text_parts).strip(), "tool_calls": tool_calls}
        else:
            # OpenAI function-calling conversion
            oai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": _upper_schema_to_lower(t.get("parameters", {})),
                    },
                }
                for t in tools
            ]
            oai_msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]
            for m in messages:
                role = m.get("role")
                if role == "user":
                    oai_msgs.append({"role": "user", "content": m.get("content") or ""})
                elif role == "assistant":
                    entry: Dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                    if m.get("tool_calls"):
                        entry["tool_calls"] = [
                            {
                                "id": tc.get("id") or f"call_{idx}",
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("arguments") or {}),
                                },
                            }
                            for idx, tc in enumerate(m["tool_calls"])
                        ]
                    oai_msgs.append(entry)
                elif role == "tool":
                    oai_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": m.get("id") or "call_0",
                            "content": m.get("content") or "",
                        }
                    )
            resp = self._post_json(
                f"{self.openai_base_url}/chat/completions",
                {
                    "model": use_model,
                    "messages": oai_msgs,
                    "tools": oai_tools,
                    "temperature": temperature,
                },
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}",
                },
            )
            choice0 = (resp.get("choices") or [{}])[0]
            finish_reason = str(choice0.get("finish_reason") or "")
            msg = choice0.get("message") or {}
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    parsed_args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_calls.append({"name": fn.get("name", ""), "arguments": parsed_args})
            usage = resp.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens") or 0)
            c_tok = int(usage.get("completion_tokens") or 0)
            result = {"content": (msg.get("content") or "").strip(), "tool_calls": tool_calls}

        with self._lock:
            self.stats["llm_api_calls"] += 1
            self.stats["prompt_tokens"] += p_tok
            self.stats["completion_tokens"] += c_tok
        if (result["content"] or result["tool_calls"]) and finish_reason not in ("MAX_TOKENS", "length"):
            self._cache_set(cache_key, result)
        return result

    def embed_texts(self, texts: Sequence[str], dim: int = 256) -> List[List[float]]:
        """Embed a batch of texts using `gemini-embedding-001` or OpenAI embeddings."""
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        for idx, t in enumerate(texts):
            norm_t = (t or " ").strip() or "empty"
            ck = hashlib.sha256(
                json.dumps(
                    {"op": "embed_v1", "model": self.embedding_model, "dim": dim, "t": norm_t},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cached = self._cache_get(ck)
            if cached is not None:
                results[idx] = cached
                with self._lock:
                    self.stats["embed_cache_hits"] += 1
            else:
                missing_indices.append(idx)

        # Fetch missing in chunks of 50
        for start in range(0, len(missing_indices), 50):
            batch_idxs = missing_indices[start : start + 50]
            batch_texts = [((texts[i] or " ").strip() or "empty")[:6000] for i in batch_idxs]
            if self.backend == "gemini":
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.embedding_model}:batchEmbedContents?key={self.gemini_api_key}"
                )
                body = {
                    "requests": [
                        {
                            "model": f"models/{self.embedding_model}",
                            "content": {"parts": [{"text": bt}]},
                            "outputDimensionality": dim,
                        }
                        for bt in batch_texts
                    ]
                }
                resp = self._post_json(url, body, {"Content-Type": "application/json"})
                embs = [e.get("values") or [] for e in (resp.get("embeddings") or [])]
            else:
                resp = self._post_json(
                    f"{self.openai_base_url}/embeddings",
                    {"model": self.embedding_model, "input": batch_texts},
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.openai_api_key}",
                    },
                )
                embs = [d.get("embedding") or [] for d in (resp.get("data") or [])]

            with self._lock:
                self.stats["embed_api_calls"] += 1

            for idx, vec in zip(batch_idxs, embs):
                norm_t = (texts[idx] or " ").strip() or "empty"
                ck = hashlib.sha256(
                    json.dumps(
                        {"op": "embed_v1", "model": self.embedding_model, "dim": dim, "t": norm_t},
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                results[idx] = vec
                self._cache_set(ck, vec)

        return [r if r is not None else [0.0] * dim for r in results]


def _upper_schema_to_lower(schema: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.lower()
        elif isinstance(v, dict):
            out[k] = _upper_schema_to_lower(v)
        else:
            out[k] = v
    return out
