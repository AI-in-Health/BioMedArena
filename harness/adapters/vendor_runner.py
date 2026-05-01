"""Subprocess-based vendor runner — isolates vendor dependencies.

Each vendor lives in its own venv at `vendors/<name>/.venv/` so dependency
conflicts (old openai, conflicting torch versions) don't poison the main process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VendorSubprocessRunner:
    """Run vendor entry points in an isolated subprocess."""

    # Map adapter-side keys → actual vendor directory names
    VENDOR_DIR_MAP = {
        "geneagent": "GeneAgent",
        "genegpt": "GeneGPT",
        "genotex": "GenoTEX",
        "genomas": "GenoMAS",
        "openbio": "OpenBioLLM",
        "medagentbench": "MedAgentBench",
        "ehragent": "EhrAgent",
        "colacare": "ColaCare",
        "mdagents": "MDAgents",
        "txagent": "TxAgent",
        "agentclinic": "AgentClinic",
        "medagent_pro": "MedAgent-Pro",
        "drugagent": "drugagent",
        "prompt2pill": "Prompt-to-Pill",
    }

    def __init__(self, vendors_dir: str = "vendors"):
        self.vendors_dir = Path(vendors_dir).resolve()

    def _resolve_dir(self, vendor_name: str) -> str:
        """Normalize adapter name → actual vendor directory name."""
        key = vendor_name.lower().replace("-", "_")
        return self.VENDOR_DIR_MAP.get(key, vendor_name)

    def is_installed(self, vendor_name: str) -> bool:
        actual = self._resolve_dir(vendor_name)
        sentinel = self.vendors_dir / actual / ".venv_ready"
        return sentinel.exists()

    def get_python(self, vendor_name: str) -> Path | None:
        """Return path to vendor-specific Python interpreter, or None."""
        actual = self._resolve_dir(vendor_name)
        venv_py = self.vendors_dir / actual / ".venv" / "bin" / "python"
        if venv_py.exists():
            return venv_py
        return None

    async def run_unified(
        self,
        vendor_name: str,
        query: str,
        context: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        """Run the unified vendor handler (`harness/vendor_entries/run_vendor.py`)
        inside the vendor's venv. Works for all 12+ vendors.
        """
        if not self.is_installed(vendor_name):
            return {"status": "not_installed",
                     "error": f"Vendor {vendor_name} has no .venv_ready sentinel"}

        py = self.get_python(vendor_name)
        if py is None:
            return {"status": "not_installed", "error": "venv python not found"}

        entry_script = Path(__file__).resolve().parent.parent / "vendor_entries" / "run_vendor.py"
        if not entry_script.exists():
            return {"status": "error", "error": f"Unified entry script missing: {entry_script}"}

        args_payload = {"query": query, "context": context or {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(args_payload, f, ensure_ascii=False)
            args_file = f.name

        # Map class adapter names to vendor directory keys
        vendor_key = vendor_name.lower().replace("-", "_")

        try:
            proc = await asyncio.create_subprocess_exec(
                str(py), str(entry_script),
                "--args-file", args_file,
                "--vendor", vendor_key,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1",
                      "PYTHONIOENCODING": "utf-8"},
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"status": "timeout",
                         "error": f"vendor {vendor_name} exceeded {timeout}s"}

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")[:500]
                return {"status": "error", "error": err_msg}

            raw_out = stdout.decode("utf-8", errors="replace").strip()
            # Handler writes a single JSON object on stdout
            try:
                # Take last non-empty line (in case of warnings above)
                last = raw_out.split("\n")[-1]
                return json.loads(last)
            except json.JSONDecodeError:
                return {"status": "error",
                         "error": f"JSON parse failed: {raw_out[:200]}"}

        finally:
            try:
                os.unlink(args_file)
            except OSError:
                pass

    async def run(
        self,
        vendor_name: str,
        entry_script: str,
        args: dict[str, Any],
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Run `vendors/<name>/<entry_script>` with args as JSON input.

        Returns dict with keys:
            status: "ok" | "not_installed" | "timeout" | "error"
            output: parsed JSON from script stdout (if status=="ok")
            error: error message (if status != "ok")
        """
        if not self.is_installed(vendor_name):
            return {
                "status": "not_installed",
                "error": f"Vendor {vendor_name} not installed (no .venv_ready sentinel)",
            }

        py = self.get_python(vendor_name)
        if py is None:
            return {"status": "not_installed", "error": "venv python not found"}

        vendor_dir = self.vendors_dir / vendor_name
        script_path = vendor_dir / entry_script
        if not script_path.exists():
            return {"status": "error", "error": f"entry script not found: {script_path}"}

        # Write args to tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(args, f)
            args_file = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                str(py), str(script_path), "--args-file", args_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(vendor_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"status": "timeout", "error": f"vendor {vendor_name} exceeded {timeout}s"}

            if proc.returncode != 0:
                return {
                    "status": "error",
                    "error": stderr.decode("utf-8", errors="replace")[:500],
                }

            try:
                output = json.loads(stdout.decode("utf-8", errors="replace"))
                return {"status": "ok", "output": output}
            except json.JSONDecodeError:
                return {"status": "ok", "output": {"raw": stdout.decode("utf-8", errors="replace")}}

        finally:
            try:
                os.unlink(args_file)
            except OSError:
                pass


# Singleton
_default_runner: VendorSubprocessRunner | None = None


def get_runner() -> VendorSubprocessRunner:
    global _default_runner
    if _default_runner is None:
        _default_runner = VendorSubprocessRunner()
    return _default_runner
