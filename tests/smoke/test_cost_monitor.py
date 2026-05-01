"""Smoke tests for the cost ledger."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fresh_ledger(monkeypatch, tmp_path):
    """Redirect the ledger file to a temp path so tests don't write to
    the real data/cost_ledger.json."""
    import harness.cost_monitor as cm
    path = tmp_path / "ledger.json"
    monkeypatch.setattr(cm, "_LEDGER_PATH", path)
    cm.reset()
    return cm


def test_record_gpt4o(fresh_ledger):
    cm = fresh_ledger
    cost = cm.record("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    # 2.50 + 10.00 = 12.50
    assert abs(cost - 12.50) < 1e-6
    snap = cm.snapshot()
    assert snap["calls"] == 1
    assert abs(snap["total_usd"] - 12.50) < 1e-6


def test_record_unknown_model_zero_cost(fresh_ledger):
    cm = fresh_ledger
    cost = cm.record("some-model-not-in-pricing", 100_000, 100_000)
    assert cost == 0.0


def test_check_budget_gate(fresh_ledger):
    cm = fresh_ledger
    ok, remaining = cm.check_budget(cap_usd=500)
    assert ok is True and remaining == 500
    cm.record("gpt-4o", 1_000_000, 1_000_000)  # $12.50
    ok, remaining = cm.check_budget(cap_usd=10)
    assert ok is False
    assert remaining == 0


def test_ledger_persists_across_snapshots(fresh_ledger):
    cm = fresh_ledger
    cm.record("gemini-2.5-flash", 2_000_000, 2_000_000)  # 0.15 + 0.60 = 0.75
    snap = cm.snapshot()
    assert snap["per_model"]["gemini-2.5-flash"] == pytest.approx(0.75)
