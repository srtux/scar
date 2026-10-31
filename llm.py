"""Simple Gemini API client with a local SQLite prompt cache.

What this file does (in plain English):
1. Sends prompts to Google's Gemini REST API using only Python's standard
   library (`urllib.request`), so no heavy cloud SDK is needed.
2. Saves every `(prompt, model, settings) -> reply` pair inside a local SQLite
   file (`.cache/llm_cache.sqlite`). When you re-run the benchmarks, cached
   replies load in less than a millisecond with zero network calls or API cost.
3. Automatically spaces out live API calls (`max_rps`) and retries temporary
   network or rate-limit hiccups (HTTP 429 / 5xx).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any
import urllib.error
import urllib.request

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)


def load_api_key(required: bool = True) -> str:
    """Read the Gemini API key from environment variables without printing secrets."""
    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()
    if required and not key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")
    return key


def parse_json_payload(raw: str) -> Any:
    """Parse a JSON dictionary or list from an LLM reply, even inside ```json fences."""
    cleaned = (raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    candidates = [cleaned]
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            candidates.append(match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


class GeminiClient:
    """Minimal thread-safe Gemini REST client backed by a SQLite WAL cache."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        cache_path: str | None = None,
        max_rps: float = 18.0,
        use_cache: bool = True,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or load_api_key(required=not use_cache)
        self.api_key = resolved_key.strip()
        self.model = model
        self.max_rps = max_rps
        self.use_cache = use_cache

        self._db_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._next_request_time = 0.0

        self.live_calls = 0
        self.cache_hits = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

        repo_dir = os.path.dirname(os.path.abspath(__file__))
        default_cache = os.path.join(repo_dir, ".cache", "llm_cache.sqlite")
        self.cache_path = cache_path or default_cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _make_cache_key(
        self,
        target_model: str,
        system_instruction: str,
        prompt: str,
        json_mode: bool,
        temperature: float,
        max_output_tokens: int,
        cache_salt: str,
    ) -> str:
        """Hash all request inputs into a unique SHA-256 key for the SQLite cache."""
        envelope = {
            "v": "memory_guard_v1",
            "m": target_model,
            "s": system_instruction,
            "p": prompt,
            "j": json_mode,
            "t": temperature,
            "k": max_output_tokens,
            "x": cache_salt,
        }
        serialized = json.dumps(envelope, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _wait_for_rate_limit(self) -> None:
        """Space out live HTTP calls so we stay below `self.max_rps` requests/second."""
        if self.max_rps <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait_seconds = self._next_request_time - now
            self._next_request_time = max(now, self._next_request_time) + (1.0 / self.max_rps)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str = "",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        model: str | None = None,
        cache_salt: str = "",
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        """Return `{"text": ..., "prompt_tokens": ..., "output_tokens": ...}`."""
        target_model = model or self.model
        allow_cache = self.use_cache if use_cache is None else use_cache
        cache_key = self._make_cache_key(
            target_model=target_model,
            system_instruction=system_instruction,
            prompt=prompt,
            json_mode=json_mode,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_salt=cache_salt,
        )

        # Step 1: Return the saved reply immediately if this exact prompt is cached.
        if allow_cache:
            with self._db_lock, sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                row = conn.execute(
                    "SELECT value FROM cache WHERE key = ?",
                    (cache_key,),
                ).fetchone()
                if row is not None:
                    self.cache_hits += 1
                    return json.loads(row[0])

        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment.")

        # Step 2: Build the Gemini REST JSON request body.
        gen_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        request_body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }
        if system_instruction:
            request_body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        payload_bytes = json.dumps(request_body).encode("utf-8")
        last_error = "unknown error"

        # Step 3: Send the HTTP request, retrying with gentle backoff on transient errors.
        for retry_delay in (0.0, 1.5, 4.0, 9.0, 18.0, 32.0):
            if retry_delay > 0:
                time.sleep(retry_delay)
            self._wait_for_rate_limit()

            url = API_URL.format(model=target_model, key=self.api_key)
            request = urllib.request.Request(
                url,
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=65) as response:
                    raw_response = json.loads(response.read().decode("utf-8"))

                candidates = raw_response.get("candidates") or []
                usage = raw_response.get("usageMetadata") or {}

                # Collect visible reply text (ignoring any internal reasoning parts).
                text_parts: list[str] = []
                for candidate in candidates:
                    parts = (candidate.get("content") or {}).get("parts") or []
                    for part in parts:
                        if not part.get("thought") and "text" in part:
                            text_parts.append(part["text"])

                result = {
                    "text": "".join(text_parts).strip(),
                    "prompt_tokens": int(usage.get("promptTokenCount") or 0),
                    "output_tokens": int(usage.get("candidatesTokenCount") or 0),
                }

                # Step 4: Save complete replies to the SQLite cache for future runs.
                finish_reason = str(candidates[0].get("finishReason") or "") if candidates else ""
                with self._db_lock:
                    self.live_calls += 1
                    self.total_prompt_tokens += result["prompt_tokens"]
                    self.total_output_tokens += result["output_tokens"]
                    if allow_cache and result["text"] and finish_reason != "MAX_TOKENS":
                        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                            conn.execute(
                                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                                (cache_key, json.dumps(result, sort_keys=True)),
                            )
                return result

            except urllib.error.HTTPError as err:
                raw_detail = err.read().decode("utf-8", errors="replace")
                safe_detail = raw_detail.replace(self.api_key, "[REDACTED]")
                last_error = f"HTTP {err.code}: {safe_detail[:600]}"
                is_transient = err.code in (429, 500, 502, 503, 504) or (
                    err.code == 400 and "dependency" in safe_detail.lower()
                )
                if not is_transient:
                    raise RuntimeError(last_error) from None
            except Exception as err:
                last_error = str(err).replace(self.api_key, "[REDACTED]")

        raise RuntimeError(f"Gemini request failed after retries: {last_error}")
