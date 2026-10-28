"""Tiny Gemini REST client with a local SQLite cache, plus token-budget helpers.

Only the Python standard library is used. Set GEMINI_API_KEY in your environment.
Every response is cached in .cache/llm_cache.sqlite, so re-running an experiment
replays identical answers instantly and for free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# Benchmark answer keys. They are removed from every chat before memory is built.
ORACLE_FORBIDDEN_KEYS = frozenset({
    "reward_info", "reward", "feedback", "evaluation_criteria", "nl_assertions", "action_checks",
    "db_check", "env_assertions", "communicate_checks", "reward_breakdown",
})

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d{1,3}|[^\sA-Za-z\d]")


def count_tokens(text: str) -> int:
    """Approximate subword token count (words, 3-digit chunks, punctuation, long-word pieces)."""
    pieces = _TOKEN_RE.findall(text or "")
    return len(pieces) + sum(max(0, (len(p) - 5) // 4) for p in pieces if p.isalpha())


def pack_items_within_budget(items: Iterable[str], max_tokens: int, header: str = "", separator: str = "\n") -> str:
    """Greedily keep items, in order, while the whole text stays within max_tokens."""
    prefix = f"{header}\n" if header else ""
    if max_tokens <= 0 or (prefix and count_tokens(prefix) >= max_tokens):
        return ""
    chosen: List[str] = []
    for item in items:
        item = item.strip()
        if item and count_tokens(prefix + separator.join(chosen + [item])) <= max_tokens:
            chosen.append(item)
    return prefix + separator.join(chosen) if chosen else ""


def strip_oracle_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a chat record without any benchmark answer-key fields (top level and one level down)."""
    clean: Dict[str, Any] = {}
    for key, value in record.items():
        if key in ORACLE_FORBIDDEN_KEYS:
            continue
        clean[key] = {k: v for k, v in value.items() if k not in ORACLE_FORBIDDEN_KEYS} if isinstance(value, dict) else value
    return clean


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse the first JSON object in a model reply, tolerating code fences and extra text."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _messages_to_gemini_contents(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn a chat history (user, assistant, tool) into alternating Gemini user/model turns.

    Earlier tool calls and tool results become plain text blocks, so a history written by
    another model can be replayed while the live turn still uses native function calling.
    """
    names_by_id: Dict[str, str] = {}
    turns: List[Dict[str, Any]] = []
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "assistant":
            blocks = [content] if content else []
            for tc in m.get("tool_calls") or []:
                name, args = str(tc.get("name") or "unknown_tool"), tc.get("arguments")
                if tc.get("id"):
                    names_by_id[str(tc["id"])] = name
                blocks.append(f"[Prior Tool Call: {name}({args if isinstance(args, str) else json.dumps(args or {}, sort_keys=True)})]")
            if blocks:
                turns.append({"role": "model", "text": "\n".join(blocks)})
        elif role == "user" and content:
            turns.append({"role": "user", "text": content})
        elif role == "tool":
            name = m.get("tool_name") or names_by_id.get(str(m.get("id") or m.get("tool_call_id") or ""), "tool_result")
            turns.append({"role": "user", "text": f"[Tool Output ({name}){' [ERROR]' if m.get('error') else ''}]: {content}"})
    merged: List[Dict[str, Any]] = []
    for t in turns:
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["parts"].append({"text": t["text"]})
        else:
            merged.append({"role": t["role"], "parts": [{"text": t["text"]}]})
    if merged and merged[0]["role"] == "model":
        merged.insert(0, {"role": "user", "parts": [{"text": "[Customer connected to support chat]"}]})
    return merged or [{"role": "user", "parts": [{"text": "Hello"}]}]


class LLMClient:
    """Thread-safe Gemini client: cache lookup first, then a rate-limited REST call with retries."""

    def __init__(self, default_model: Optional[str] = None, fast_model: Optional[str] = None,
                 cache_path: Optional[str] = None, use_cache: bool = True) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your environment to run the experiments.")
        self.default_model = default_model or os.environ.get("ESM_LLM_MODEL", "gemini-3.8-flash")
        self.fast_model = fast_model or os.environ.get("ESM_FAST_MODEL", "gemini-3.5-flash-lite")
        self.use_cache = use_cache
        self.max_rps = float(os.environ.get("ESM_MAX_RPS", "14"))
        self._lock, self._rate_lock, self._next_ts = threading.Lock(), threading.Lock(), 0.0
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.cache_path = cache_path or os.path.join(repo_root, ".cache", "llm_cache.sqlite")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _cache(self, key: str, value: Any = None) -> Any:
        """Read the cached value for key, or store value when one is given."""
        if not self.use_cache:
            return None
        with self._lock, sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            if value is None:
                row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
                return json.loads(row[0]) if row else None
            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))
        return None

    def _call(self, model: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST to Gemini with client-side rate limiting and retries on 429/5xx errors."""
        data, last_err = json.dumps(body).encode("utf-8"), None
        for delay in (0, 2, 5, 10, 18, 30, 45):
            time.sleep(delay)
            if self.max_rps > 0:
                with self._rate_lock:
                    now = time.monotonic()
                    wait, self._next_ts = self._next_ts - now, max(now, self._next_ts) + 1.0 / self.max_rps
                time.sleep(max(0.0, wait))
            req = urllib.request.Request(API_URL.format(model=model, key=self.api_key), data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {e.code}: {detail[:1200]}")
                # Gemini sometimes reports a failing backend dependency as HTTP 400; that case is safe to retry.
                if e.code not in (429, 500, 502, 503, 504) and not (e.code == 400 and "dependency" in detail):
                    raise last_err from e
            except Exception as e:  # network hiccups: retry
                last_err = e
        raise RuntimeError(f"Gemini request failed after retries: {last_err}")

    @staticmethod
    def _config(model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if "gemini-3" in model or "gemini-4" in model:
            cfg["thinkingConfig"] = {"thinkingLevel": "LOW"}
        return cfg

    @staticmethod
    def _parts(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
        cands = resp.get("candidates") or []
        return [p for c in cands for p in (c.get("content") or {}).get("parts") or [] if not p.get("thought")]

    @staticmethod
    def _finished(resp: Dict[str, Any]) -> bool:
        cands = resp.get("candidates") or []
        return str(cands[0].get("finishReason") or "") != "MAX_TOKENS" if cands else False

    def generate_text(self, prompt: str, system_instruction: str = "", model: Optional[str] = None,
                      json_mode: bool = False, temperature: float = 0.0, max_output_tokens: int = 1024) -> str:
        """One-shot text (or JSON string) generation."""
        model = model or self.default_model
        key = hashlib.sha256(json.dumps({"op": "generate_text_v1", "backend": "gemini", "model": model,
                                         "sys": system_instruction, "prompt": prompt, "json_mode": json_mode,
                                         "temp": temperature, "max_tok": max_output_tokens}, sort_keys=True).encode()).hexdigest()
        hit = self._cache(key)
        if hit is not None:
            return str(hit.get("text", ""))
        cfg = self._config(model, temperature, max(max_output_tokens, 4096))
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body: Dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        resp = self._call(model, body)
        text = "".join(p["text"] for p in self._parts(resp) if "text" in p).strip()
        if text and self._finished(resp):
            self._cache(key, {"text": text})
        return text

    def generate_json(self, prompt: str, system_instruction: str = "", model: Optional[str] = None,
                      max_output_tokens: int = 1200) -> Dict[str, Any]:
        return parse_json_object(self.generate_text(prompt, system_instruction, model, json_mode=True,
                                                    max_output_tokens=max_output_tokens))

    def generate_agent_step(self, messages: Sequence[Dict[str, Any]], system_instruction: str,
                            tools: Sequence[Dict[str, Any]], model: Optional[str] = None, temperature: float = 0.0) -> Dict[str, Any]:
        """Ask the agent for its next move (text and/or native tool calls) given the chat so far."""
        model = model or self.default_model
        key = hashlib.sha256(json.dumps({"op": "agent_step_v1", "backend": "gemini", "model": model,
                                         "sys": system_instruction, "messages": messages, "tools": tools,
                                         "temp": temperature}, sort_keys=True).encode()).hexdigest()
        hit = self._cache(key)
        if hit is not None:
            return dict(hit)
        resp = self._call(model, {"systemInstruction": {"parts": [{"text": system_instruction}]},
                                  "contents": _messages_to_gemini_contents(messages),
                                  "tools": [{"functionDeclarations": list(tools)}],
                                  "generationConfig": self._config(model, temperature, 4096)})
        parts = self._parts(resp)
        result = {"content": "".join(p["text"] for p in parts if "text" in p).strip(),
                  "tool_calls": [{"name": p["functionCall"].get("name", ""), "arguments": p["functionCall"].get("args") or {}}
                                 for p in parts if "functionCall" in p]}
        if (result["content"] or result["tool_calls"]) and self._finished(resp):
            self._cache(key, result)
        return result
