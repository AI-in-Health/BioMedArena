"""Per-benchmark harness configs.

Importing this package side-effect-registers every entry in `registry.py`
into `harness.benchmark_config.BENCHMARK_CONFIGS`.
"""
from __future__ import annotations

from harness.benchmark_configs import registry  # noqa: F401
