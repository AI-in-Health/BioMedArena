"""Shared mixin: call unified vendor subprocess with graceful fallback.

Vendor adapters inherit this and call `self._try_vendor(query, context)` —
if subprocess succeeds (status=ok), returns the vendor's answer dict.
If subprocess returns fallback/error/timeout, returns None so the adapter
can execute its own native reimplementation.
"""

from __future__ import annotations

from typing import Any

from harness.adapters.vendor_runner import get_runner


async def try_vendor(
    vendor_name: str,
    query: str,
    context: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """Attempt vendor subprocess call. Returns the result dict on ok, None otherwise."""
    runner = get_runner()
    try:
        result = await runner.run_unified(vendor_name, query, context, timeout=timeout)
    except Exception:
        return None

    if result.get("status") == "ok" and result.get("answer"):
        return result
    return None
