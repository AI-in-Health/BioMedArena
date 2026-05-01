"""Disk-backed cache for retrieval API calls (SQLite via diskcache)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from diskcache import Cache
    _HAVE_DISKCACHE = True
except ImportError:
    _HAVE_DISKCACHE = False


class RetrievalCache:
    """24h TTL cache keyed by (host, path, sorted params)."""

    def __init__(self, cache_dir: str = "data/cache/retrieval", ttl_seconds: int = 86400):
        self.ttl = ttl_seconds
        self._cache: Any = None
        if _HAVE_DISKCACHE:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(cache_dir))

    @staticmethod
    def _key(host: str, path: str, params: dict | None = None) -> str:
        params_str = json.dumps(params or {}, sort_keys=True)
        raw = f"{host}|{path}|{params_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, host: str, path: str, params: dict | None = None) -> Any:
        if self._cache is None:
            return None
        return self._cache.get(self._key(host, path, params))

    def set(self, host: str, path: str, params: dict | None, value: Any) -> None:
        if self._cache is None:
            return
        self._cache.set(self._key(host, path, params), value, expire=self.ttl)


# Singleton default cache
_default_cache: RetrievalCache | None = None


def get_cache() -> RetrievalCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = RetrievalCache()
    return _default_cache
