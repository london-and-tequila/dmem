"""
LLM disk cache + rate-limit retry for multi-process experiment runs.

Components
----------
LLMCache        – SQLite-backed persistent cache (WAL mode, fork-safe)
retry_on_rate_limit – exponential-backoff decorator for transient API errors
get_global_cache / init_global_cache – per-process singleton management
"""

import hashlib
import json
import os
import sqlite3
import time
import functools
from typing import Optional


# ---------------------------------------------------------------------------
# A. SQLite persistent cache
# ---------------------------------------------------------------------------

class LLMCache:
    """Fork-safe SQLite cache.  Each get/put opens its own connection so the
    object can survive ``os.fork()`` in multiprocessing pools."""

    def __init__(self, db_path: str = ".llm_cache.db"):
        self.db_path = os.path.abspath(db_path)
        self._ensure_table()

    # -- internal helpers ---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_table(self):
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key     TEXT PRIMARY KEY,
                    response      TEXT NOT NULL,
                    prompt_preview TEXT,
                    model         TEXT,
                    created_at    REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    # -- public API ---------------------------------------------------------

    @staticmethod
    def make_key(prompt: str, model: str, temperature: float,
                 response_format: dict) -> str:
        blob = json.dumps(
            {"prompt": prompt, "model": model,
             "temperature": temperature, "response_format": response_format},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def put(self, key: str, response: str, prompt: str = "",
            model: str = ""):
        if not response:
            return
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO llm_cache
                   (cache_key, response, prompt_preview, model, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, response, prompt[:200], model, time.time()),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# B. Exponential-backoff retry decorator
# ---------------------------------------------------------------------------

def retry_on_rate_limit(max_retries: int = 5, base_delay: float = 2.0):
    """Decorator: retry on transient API errors, re-raise others as
    pickle-safe ``RuntimeError``."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if _is_retryable(exc):
                        delay = _backoff_delay(exc, attempt, base_delay)
                        print(f"[retry {attempt+1}/{max_retries}] "
                              f"{type(exc).__name__}: {exc} — "
                              f"waiting {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        raise RuntimeError(
                            f"{type(exc).__name__}: {exc}"
                        ) from None
            # exhausted retries
            raise RuntimeError(
                f"Exhausted {max_retries} retries. "
                f"Last error: {type(last_exc).__name__}: {last_exc}"
            ) from None
        return wrapper
    return decorator


def _is_retryable(exc: Exception) -> bool:
    """Return True for 429, 5xx, timeout, and connection errors."""
    cls_name = type(exc).__name__

    # openai / httpx status errors
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        status = int(status)
        if status == 429 or 500 <= status < 600:
            return True

    # requests library
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        code = exc.response.status_code
        if code == 429 or 500 <= code < 600:
            return True

    # Timeout / connection
    for keyword in ("Timeout", "timeout", "Connection", "connection",
                    "APIConnectionError", "APITimeoutError"):
        if keyword in cls_name or keyword in str(exc):
            return True

    # openai RateLimitError (may not have .status_code)
    if "RateLimitError" in cls_name or "RateLimit" in cls_name:
        return True

    return False


def _backoff_delay(exc: Exception, attempt: int, base_delay: float) -> float:
    """Compute wait time. Honour Retry-After header when present."""
    # Check Retry-After
    retry_after = None
    headers = None
    if hasattr(exc, "headers"):
        headers = exc.headers
    elif hasattr(exc, "response") and hasattr(exc.response, "headers"):
        headers = exc.response.headers
    if headers:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")

    if retry_after is not None:
        try:
            return max(float(retry_after), 1.0)
        except (ValueError, TypeError):
            pass

    # Exponential backoff, 429 gets a 10s floor
    delay = base_delay * (2 ** attempt)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429 or "RateLimit" in type(exc).__name__:
        delay = max(delay, 10.0)
    return min(delay, 120.0)


# ---------------------------------------------------------------------------
# C. Global cache singleton (per-process)
# ---------------------------------------------------------------------------

_global_cache: Optional[LLMCache] = None
_global_cache_path: Optional[str] = None
_cache_disabled: bool = False


def init_global_cache(db_path: str = None, disabled: bool = False):
    """Call once in the main process to configure the cache path."""
    global _global_cache_path, _cache_disabled, _global_cache
    _cache_disabled = disabled
    if db_path is not None:
        _global_cache_path = os.path.abspath(db_path)
    _global_cache = None  # reset so next get re-creates


def get_global_cache() -> Optional[LLMCache]:
    """Return per-process cache instance, or None if caching is disabled."""
    global _global_cache
    if _cache_disabled:
        return None
    if _global_cache is None:
        path = _global_cache_path or os.path.join(
            os.path.dirname(__file__), ".llm_cache.db"
        )
        _global_cache = LLMCache(path)
    return _global_cache
