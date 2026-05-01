"""Per-host async rate limiter using token bucket (aiolimiter)."""

from __future__ import annotations

from typing import Any

try:
    from aiolimiter import AsyncLimiter
    _HAVE_LIMITER = True
except ImportError:
    _HAVE_LIMITER = False


# Per-host rate limits (requests per second)
DEFAULT_LIMITS: dict[str, float] = {
    "rxnav.nlm.nih.gov": 15,           # Official: 20/s
    "api.fda.gov": 4,                   # Official: 240/min
    "dailymed.nlm.nih.gov": 5,
    "eutils.ncbi.nlm.nih.gov": 3,      # Without API key; 10 with
    "api.omim.org": 2,
    "api.orphadata.com": 5,
    "www.orphadata.org": 5,
    "connect.medlineplus.gov": 3,
    "wsearch.nlm.nih.gov": 3,
}

_limiters: dict[str, Any] = {}


def get_limiter(host: str) -> Any:
    """Get or create a rate limiter for a host. Returns None if aiolimiter not installed."""
    if not _HAVE_LIMITER:
        return None
    if host not in _limiters:
        rate = DEFAULT_LIMITS.get(host, 5)
        _limiters[host] = AsyncLimiter(max_rate=rate, time_period=1.0)
    return _limiters[host]


class _NullLimiter:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False


def acquire(host: str):
    """Return a context manager for rate limiting (or null if disabled)."""
    lim = get_limiter(host)
    return lim if lim is not None else _NullLimiter()
