"""Gemini client using the official `google-genai` SDK, `.env` config, and SQLite cache.

What this file does (in plain English):
1. Loads configuration automatically from `.env` (or shell environment variables),
   supporting either a standard `GEMINI_API_KEY` (Google AI Studio) or Google Cloud
   Vertex AI (`GOOGLE_CLOUD_PROJECT`), plus optional `GEMINI_MODEL` overrides.
2. Uses `tenacity` for automatic exponential-backoff retries on temporary rate limits
   or network hiccups.
3. Caches every `(model, system_instruction, prompt, settings) -> reply` inside a local
   SQLite database (`.cache/llm_cache.sqlite`) so re-running the benchmark offline
   takes milliseconds and zero API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

DEFAULT_MODEL = "gemini-3.5-flash-lite"

_PLACEHOLDER_VALUES = frozenset({
    "your_gemini_api_key_here",
    "your-api-key",
    "your-gcp-project-id",
    "change_me",
    "changeme",
})


def load_dotenv_file(
    dotenv_path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load `KEY=VALUE` pairs from `.env` into `os.environ` without external deps."""
    if dotenv_path is not None:
        path = Path(dotenv_path)
    else:
        env_override = os.environ.get("MEMORY_GUARD_DOTENV_PATH")
        if env_override == "":
            return {}
        path = (
            Path(env_override)
            if env_override is not None
            else Path(__file__).resolve().parent / ".env"
        )

    if not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _clean_env_value(raw: str | None) -> str:
    """Strip whitespace and ignore template placeholder strings from `.env.example`."""
    cleaned = (raw or "").strip()
    if cleaned.lower() in _PLACEHOLDER_VALUES:
        return ""
    return cleaned


def resolve_model(model: str | None = None) -> str:
    """Resolve explicit `model`, `GEMINI_MODEL` from env/`.env`, or `DEFAULT_MODEL`."""
    load_dotenv_file()
    explicit = _clean_env_value(model)
    if explicit:
        return explicit
    from_env = _clean_env_value(os.environ.get("GEMINI_MODEL"))
    return from_env or DEFAULT_MODEL


def load_api_key(required: bool = True) -> str:
    """Read `GEMINI_API_KEY` or `GOOGLE_API_KEY` from environment variables or `.env`."""
    load_dotenv_file()
    key = _clean_env_value(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    project = _clean_env_value(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if required and not key and not project:
        raise RuntimeError(
            "Missing Gemini credentials: copy .env.example to .env and set "
            "GEMINI_API_KEY, or set GOOGLE_CLOUD_PROJECT."
        )
    return key


def has_live_credentials() -> bool:
    """Return True if a usable `GEMINI_API_KEY` or `GOOGLE_CLOUD_PROJECT` is configured."""
    load_dotenv_file()
    return bool(
        load_api_key(required=False)
        or _clean_env_value(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    )


def parse_json_payload(raw: str) -> Any:
    """Parse a JSON dict or list from an LLM reply, stripping optional markdown fences."""
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


def _is_transient_error(exc: BaseException) -> bool:
    """Retry on HTTP 429 rate limits, 5xx server errors, or transient socket timeouts."""
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (429, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class GeminiClient:
    """Thread-safe `google-genai` wrapper with a local SQLite WAL prompt cache."""

    def __init__(
        self,
        model: str | None = None,
        cache_path: str | None = None,
        max_rps: float = 18.0,
        use_cache: bool = True,
        api_key: str | None = None,
    ) -> None:
        load_dotenv_file()
        self.api_key = _clean_env_value(api_key) or load_api_key(required=not use_cache)
        self.project = _clean_env_value(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        self.location = _clean_env_value(os.environ.get("GOOGLE_CLOUD_LOCATION")) or "global"
        self.model = resolve_model(model)
        self.max_rps = max_rps
        self.use_cache = use_cache

        self._sdk_client: genai.Client | None = None
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

    def _get_sdk_client(self) -> genai.Client:
        """Lazily initialize the `google-genai` client only when a live call is needed."""
        if self._sdk_client is not None:
            return self._sdk_client

        if self.api_key:
            self._sdk_client = genai.Client(api_key=self.api_key)
            return self._sdk_client

        if self.project:
            impersonate_sa = (
                os.environ.get("GOOGLE_IMPERSONATE_SERVICE_ACCOUNT")
                or os.environ.get("VERTEX_IMPERSONATE_SERVICE_ACCOUNT")
                or ""
            ).strip()
            credentials = None
            if impersonate_sa:
                import google.auth
                from google.auth import impersonated_credentials

                source_creds, _ = google.auth.default()
                credentials = impersonated_credentials.Credentials(
                    source_credentials=source_creds,
                    target_principal=impersonate_sa,
                    target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            self._sdk_client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                credentials=credentials,
            )
            return self._sdk_client

        raise RuntimeError(
            "Missing Gemini credentials: copy .env.example to .env and set "
            "GEMINI_API_KEY, or set GOOGLE_CLOUD_PROJECT."
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
        """Hash all request parameters into a SHA-256 cache lookup key."""
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
        """Space out live calls so concurrent workers stay under `self.max_rps`."""
        if self.max_rps <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait_seconds = self._next_request_time - now
            self._next_request_time = max(now, self._next_request_time) + (1.0 / self.max_rps)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=1.5, max=30.0),
        retry=retry_if_exception(_is_transient_error),
        reraise=True,
    )
    def _call_model(
        self,
        target_model: str,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Execute one live `google-genai` call with automatic `tenacity` retries."""
        self._wait_for_rate_limit()
        return self._get_sdk_client().models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )

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

        # Step 1: Return cached reply immediately if present.
        if allow_cache:
            with self._db_lock, sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                row = conn.execute(
                    "SELECT value FROM cache WHERE key = ?",
                    (cache_key,),
                ).fetchone()
                if row is not None:
                    self.cache_hits += 1
                    return json.loads(row[0])

        # Step 2: Configure and invoke `google-genai`.
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction or None,
            response_mime_type="application/json" if json_mode else None,
        )
        response = self._call_model(target_model, prompt, config)

        usage = response.usage_metadata
        result = {
            "text": (response.text or "").strip(),
            "prompt_tokens": int((usage.prompt_token_count if usage else 0) or 0),
            "output_tokens": int((usage.candidates_token_count if usage else 0) or 0),
        }

        finish_reason = ""
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = str(response.candidates[0].finish_reason)

        # Step 3: Cache non-truncated replies for deterministic offline replay.
        with self._db_lock:
            self.live_calls += 1
            self.total_prompt_tokens += result["prompt_tokens"]
            self.total_output_tokens += result["output_tokens"]
            if allow_cache and result["text"] and "MAX_TOKENS" not in finish_reason:
                with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                        (cache_key, json.dumps(result, sort_keys=True)),
                    )
        return result
