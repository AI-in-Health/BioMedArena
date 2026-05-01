"""Smoke tests for PyTDC — vendor-isolated subprocess integration.

These tests are slow (first call triggers full PyTDC import in the vendor
venv, ~10-30s cold start). Marked `slow` so default pytest can skip them.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from harness.adapters.pytdc_adapter import (
    PyTDCAdapter, PyTDCServer, get_server, _VENV_PY, _ENTRY,
)


pytestmark = pytest.mark.asyncio

# Skip cleanly if the vendor venv isn't set up (allows CI to skip)
_available = _VENV_PY.exists() and _ENTRY.exists()


@pytest.fixture(autouse=True)
async def _cleanup_shared_server():
    """Ensure the shared PyTDC server is shut down after each test."""
    yield
    import harness.adapters.pytdc_adapter as mod
    if mod._SHARED_SERVER is not None:
        try:
            await mod._SHARED_SERVER.stop()
        except Exception:
            pass
        mod._SHARED_SERVER = None


@pytest.mark.skipif(not _available, reason="PyTDC vendor venv not installed")
async def test_subprocess_ping():
    srv = PyTDCServer(startup_timeout=60)
    t0 = time.monotonic()
    await srv.start()
    startup_s = time.monotonic() - t0
    assert startup_s < 30, f"cold start too slow: {startup_s}s"
    resp = await srv.call("ping", timeout=10)
    assert resp["ok"] is True
    await srv.stop()


@pytest.mark.skipif(not _available, reason="PyTDC vendor venv not installed")
async def test_persistence_second_call_fast():
    """Proves subprocess persists: 2nd call should be <<1st call."""
    srv = PyTDCServer(startup_timeout=60)
    await srv.start()
    t1 = time.monotonic()
    await srv.call("ping", timeout=10)
    first = time.monotonic() - t1
    t2 = time.monotonic()
    await srv.call("ping", timeout=10)
    second = time.monotonic() - t2
    await srv.stop()
    # ping is trivial both times but confirms no restart happened
    assert second < 1.0, f"2nd call suspiciously slow: {second}s"


@pytest.mark.skipif(not _available, reason="PyTDC vendor venv not installed")
async def test_admet_lookup_aspirin():
    """Load a small ADMET dataset and check the adapter can query it."""
    adapter = PyTDCAdapter()
    assert adapter.available

    resp = await adapter.admet_predict(
        "CC(=O)Oc1ccccc1C(=O)O",
        endpoints=["Caco2_Wang"],
    )
    assert resp["ok"] is True, f"admet_predict failed: {resp}"
    preds = resp.get("predictions", {})
    assert "Caco2_Wang" in preds
    # Whether aspirin is in Caco2_Wang dataset is dataset-specific; value
    # may be a float or None — we just require the key to exist
    val = preds["Caco2_Wang"]
    assert val is None or isinstance(val, (int, float))


@pytest.mark.skipif(not _available, reason="PyTDC vendor venv not installed")
async def test_dataset_sample():
    adapter = PyTDCAdapter()
    resp = await adapter.load_dataset_sample("Caco2_Wang", n=3)
    assert resp["ok"] is True
    assert len(resp.get("sample", [])) == 3
    assert resp.get("total_rows", 0) > 100


def test_adapter_registry_has_pytdc():
    from harness.adapters import ADAPTER_REGISTRY
    assert "PyTDCAdapter" in ADAPTER_REGISTRY


def test_pytdc_tools_in_specs():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = [t["function"]["name"] for t in TOOL_SPECS]
    for expected in [
        "tdc_admet_lookup", "tdc_load_dataset_sample",
        "tdc_molecule_generation_sample",
    ]:
        assert expected in names
