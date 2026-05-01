"""PyTDC adapter — vendor-isolated persistent subprocess.

PyTDC (Therapeutics Data Commons) requires several older versions of
numpy/pandas/datasets/rdkit that would downgrade our main venv. We
therefore run it in `vendors/pytdc/.venv311` and talk to it via a
long-lived subprocess server (`vendors/pytdc/entry.py`) using JSON
over stdin/stdout.

First call pays the 10-30s cold-start cost (PyTDC imports + dataset
loads). Subsequent calls are fast (<1s typical) because the server
caches `ADME` datasets in RAM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase

logger = logging.getLogger(__name__)


_VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendors" / "pytdc"
_VENV_PY = _VENDOR_DIR / ".venv311" / "bin" / "python"
_ENTRY = _VENDOR_DIR / "entry.py"


class PyTDCServer:
    """Long-lived subprocess with line-delimited JSON over stdin/stdout."""

    def __init__(self, startup_timeout: int = 45):
        self.startup_timeout = startup_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._ready = False

    async def start(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        if not _VENV_PY.exists() or not _ENTRY.exists():
            raise FileNotFoundError(
                f"PyTDC vendor not installed: {_VENV_PY} or {_ENTRY} missing"
            )
        self._proc = await asyncio.create_subprocess_exec(
            str(_VENV_PY),
            str(_ENTRY),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_VENDOR_DIR),
        )
        # Wait for _startup announce
        try:
            line = await asyncio.wait_for(
                self._proc.stdout.readline(), timeout=self.startup_timeout
            )
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise TimeoutError(
                f"PyTDC subprocess did not announce ready within {self.startup_timeout}s"
            ) from exc
        payload = json.loads(line.decode("utf-8").strip())
        if not payload.get("ready"):
            raise RuntimeError(f"PyTDC bad startup: {payload}")
        self._ready = True

    async def call(self, cmd: str, timeout: int = 60, **kwargs: Any) -> dict[str, Any]:
        if not self._ready:
            await self.start()
        assert self._proc is not None and self._proc.stdin is not None
        async with self._lock:
            cmd_id = uuid.uuid4().hex[:8]
            payload = {"id": cmd_id, "cmd": cmd, **kwargs}
            try:
                self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
                await self._proc.stdin.drain()
            except BrokenPipeError:
                # Process died; restart and retry once
                self._ready = False
                await self.start()
                return await self.call(cmd, timeout=timeout, **kwargs)

            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=timeout
                )
            except asyncio.TimeoutError:
                return {"id": cmd_id, "ok": False, "error": f"timeout after {timeout}s"}

            if not line:
                self._ready = False
                return {"id": cmd_id, "ok": False, "error": "EOF from subprocess"}
            return json.loads(line.decode("utf-8").strip())

    async def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.returncode is None:
            try:
                # Graceful shutdown
                assert self._proc.stdin is not None
                self._proc.stdin.write(b'{"id":"_stop","cmd":"shutdown"}\n')
                await self._proc.stdin.drain()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        self._ready = False


# Module-level shared instance (one PyTDC subprocess per Python process)
_SHARED_SERVER: PyTDCServer | None = None


async def get_server() -> PyTDCServer:
    global _SHARED_SERVER
    if _SHARED_SERVER is None:
        _SHARED_SERVER = PyTDCServer()
    if not _SHARED_SERVER._ready:
        await _SHARED_SERVER.start()
    return _SHARED_SERVER


class PyTDCAdapter(AdapterBase):
    name = "pytdc"
    modality = "drug"
    description = (
        "Therapeutics Data Commons (PyTDC) access: ADMET endpoint lookup, "
        "dataset sampling, and molecule generation examples. Runs in an "
        "isolated vendor venv via a persistent subprocess server."
    )

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        # Availability: check the venv exists; real startup is lazy
        if not _VENV_PY.exists():
            self.mark_unavailable(
                "PyTDC vendor venv missing. "
                "Run: uv venv --python 3.11 vendors/pytdc/.venv311 && "
                "uv pip install --python vendors/pytdc/.venv311/bin/python PyTDC"
            )

    def capabilities(self) -> list[str]:
        return [
            "admet_dataset_lookup",
            "drug_property_sampling",
            "tdc_dataset_access",
            "molecule_generation_sampling",
        ]

    async def admet_predict(
        self, smiles: str, endpoints: list[str] | None = None
    ) -> dict[str, Any]:
        srv = await get_server()
        return await srv.call(
            "admet_predict",
            smiles=smiles,
            endpoints=endpoints or ["Caco2_Wang"],
            timeout=120,
        )

    async def load_dataset_sample(
        self, name: str, n: int = 5
    ) -> dict[str, Any]:
        srv = await get_server()
        return await srv.call(
            "load_dataset_sample", subset=name, n=n, timeout=120,
        )

    async def mol_generation_sample(self, n: int = 5) -> dict[str, Any]:
        srv = await get_server()
        return await srv.call("mol_generation_sample", n=n, timeout=120)

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.available:
            return self.result(answer=self.unavailable_reason, confidence=0.0)

        ctx = context or {}
        op = ctx.get("operation", "admet_predict")
        try:
            if op == "admet_predict":
                smiles = ctx.get("smiles", "")
                endpoints = ctx.get("endpoints") or ["Caco2_Wang"]
                if not smiles:
                    return self.result(
                        answer="PyTDC admet_predict needs context['smiles'].",
                        confidence=0.1,
                    )
                resp = await self.admet_predict(smiles, endpoints)
                if resp.get("ok"):
                    preds = resp.get("predictions", {})
                    lines = [f"{k}: {v}" for k, v in preds.items() if not k.startswith("_")]
                    return self.result(
                        answer="ADMET lookup:\n" + "\n".join(lines),
                        evidence=[f"TDC endpoints queried: {', '.join(endpoints)}"],
                        confidence=0.85 if any(v is not None for v in preds.values()) else 0.3,
                        raw=resp,
                    )
            elif op == "load_dataset_sample":
                name = ctx.get("name", "Caco2_Wang")
                n = int(ctx.get("n", 5))
                resp = await self.load_dataset_sample(name, n)
                if resp.get("ok"):
                    sample = resp.get("sample", [])
                    return self.result(
                        answer=f"TDC dataset '{name}' first {len(sample)} rows (total: {resp.get('total_rows')}):\n"
                                + "\n".join(str(r) for r in sample[:3]),
                        confidence=0.85,
                        raw=resp,
                    )
            elif op == "mol_generation_sample":
                n = int(ctx.get("n", 5))
                resp = await self.mol_generation_sample(n)
                if resp.get("ok"):
                    samples = resp.get("samples", [])
                    return self.result(
                        answer=f"TDC MolGen (ZINC) {len(samples)} sample SMILES:\n" +
                                "\n".join(samples),
                        confidence=0.85,
                        raw=resp,
                    )

            return self.result(answer=f"PyTDC error: {resp}", confidence=0.0, raw=resp)
        except Exception as exc:
            return self.result(answer=f"PyTDC adapter error: {exc}", confidence=0.0)
