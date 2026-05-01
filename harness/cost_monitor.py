"""Global cost monitor — tracks spend across benchmark runs.

Reads/writes `data/cost_ledger.json` so spend survives across Python
processes (important for multi-hour sweeps). Call `record()` after
every paid LLM call and `check_budget()` before starting a big batch.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# USD per 1M tokens — update when providers change prices. Rough as of
# 2026-04; fine for budget gating, not for invoicing.
PRICING: dict[str, tuple[float, float]] = {
    # model → (input $/1M, output $/1M)
    # Established tiers
    "gpt-4o":                  (2.50, 10.00),
    "gpt-4o-mini":             (0.15,  0.60),
    "o4-mini":                 (3.00, 12.00),
    "o3":                      (10.00, 40.00),
    "claude-sonnet-4-5":       (3.00, 15.00),
    "claude-opus-4-5":         (15.00, 75.00),
    "claude-haiku-4":          (0.80,  4.00),
    "claude-haiku-4-5":        (0.80,  4.00),
    "gemini-2.5-flash":        (0.075, 0.30),
    "gemini-2.5-pro":          (1.25, 10.00),
    "text-embedding-3-small":  (0.02,  0.00),
    # Newer Claude tiers (prices are best-effort estimates; verify at billing time)
    "claude-sonnet-4-6":       (3.00, 15.00),
    "claude-opus-4-6":         (15.00, 75.00),
}


_LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "cost_ledger.json"
_LOCK = threading.Lock()


@dataclass
class Ledger:
    total_usd: float = 0.0
    per_model: dict[str, float] = field(default_factory=dict)
    calls: int = 0
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


def _load() -> Ledger:
    if not _LEDGER_PATH.exists():
        return Ledger()
    try:
        return Ledger(**json.loads(_LEDGER_PATH.read_text()))
    except Exception:
        return Ledger()


def _save(l: Ledger) -> None:
    _LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    l.last_updated = datetime.utcnow().isoformat() + "Z"
    # Atomic write: tmp + rename so concurrent readers never see partial JSON.
    tmp = _LEDGER_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(l), indent=2))
    tmp.replace(_LEDGER_PATH)


def record(model: str, input_tokens: int, output_tokens: int) -> float:
    """Record one paid call; return its USD cost."""
    in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
    cost = input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate
    with _LOCK:
        l = _load()
        l.total_usd += cost
        l.per_model[model] = l.per_model.get(model, 0.0) + cost
        l.calls += 1
        _save(l)
    return cost


def check_budget(cap_usd: float = 500.0) -> tuple[bool, float]:
    """Return (ok, remaining_usd). ok=False when spend >= cap."""
    l = _load()
    return (l.total_usd < cap_usd, max(0.0, cap_usd - l.total_usd))


def snapshot() -> dict:
    return asdict(_load())


def reset() -> None:
    with _LOCK:
        _save(Ledger())
