#!/usr/bin/env python3
"""Persistent PyTDC subprocess server.

Launched by PyTDCAdapter on first use. Keeps PyTDC models cached in
process memory between calls so only the FIRST call pays the 10-30s
cold-start cost.

Protocol
--------
One JSON command per line on stdin; one JSON response per line on stdout.
Lines are newline-delimited UTF-8.

Commands
--------
{"id": "...", "cmd": "ping"}
    → {"id": "...", "ok": true}

{"id": "...", "cmd": "admet_predict",
 "smiles": "CC(=O)Oc1ccccc1C(=O)O",
 "endpoints": ["Caco2_Wang", "Lipophilicity_AstraZeneca"]}
    → {"id": "...", "ok": true, "predictions": {"Caco2_Wang": -4.31, ...}}

{"id": "...", "cmd": "load_dataset_sample",
 "name": "ADME", "subset": "Caco2_Wang", "n": 5}
    → {"id": "...", "ok": true, "sample": [{"drug": "...", "y": ...}, ...]}

{"id": "...", "cmd": "mol_generation_sample",
 "task": "ligand_generation", "n": 5}
    → {"id": "...", "ok": true, "samples": ["CC(=O)...", ...]}

{"id": "...", "cmd": "shutdown"}
    → {"id": "...", "ok": true}  then exits
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any


# ------------------------------------------------------------------ IO


def _write(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _ok(cmd_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": cmd_id, "ok": True, **fields}


def _err(cmd_id: str, error: str) -> dict[str, Any]:
    return {"id": cmd_id, "ok": False, "error": error[:1200]}


# --------------------------------------------------------- lazy PyTDC


class TDCCache:
    """Lazy-loaded PyTDC resources kept alive across calls."""

    def __init__(self) -> None:
        self._admet_datasets: dict[str, Any] = {}

    def get_admet_dataset(self, subset: str):
        """Load a single-instance ADMET dataset; cache it."""
        if subset in self._admet_datasets:
            return self._admet_datasets[subset]
        from tdc.single_pred import ADME  # type: ignore
        data = ADME(name=subset)
        self._admet_datasets[subset] = data
        return data


# --------------------------------------------------------- handlers


def handle_ping(cmd: dict, cache: TDCCache) -> dict:
    return _ok(cmd["id"])


def handle_admet_predict(cmd: dict, cache: TDCCache) -> dict:
    """Return known y-values for the SMILES if present in TDC datasets.

    NOTE: TDC is primarily a *dataset* library. For true prediction we'd
    need a pre-trained model which TDC ships separately. This handler
    returns the ground-truth y value if the SMILES matches an entry in
    the requested dataset, else NaN-as-None.
    """
    smiles = cmd.get("smiles", "")
    endpoints = cmd.get("endpoints") or ["Caco2_Wang"]
    preds: dict[str, float | None] = {}
    if not smiles:
        return _err(cmd["id"], "missing smiles")

    # Normalise via rdkit if available
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        canon = Chem.MolToSmiles(mol) if mol else smiles
    except Exception:
        canon = smiles

    for ep in endpoints:
        try:
            data = cache.get_admet_dataset(ep)
            df = data.get_data()
            # TDC frames use 'Drug' column for SMILES and 'Y' for endpoint
            col_drug = "Drug" if "Drug" in df.columns else df.columns[1]
            col_y = "Y" if "Y" in df.columns else df.columns[-1]
            hit = df[df[col_drug] == canon]
            if hit.empty:
                hit = df[df[col_drug] == smiles]
            preds[ep] = float(hit[col_y].iloc[0]) if not hit.empty else None
        except Exception as exc:
            preds[ep] = None
            preds[f"_{ep}_error"] = str(exc)[:200]  # type: ignore

    return _ok(cmd["id"], predictions=preds, canonical=canon)


def handle_load_dataset_sample(cmd: dict, cache: TDCCache) -> dict:
    """Return first N rows of a TDC dataset for ad-hoc inspection."""
    subset = cmd.get("subset") or cmd.get("name") or "Caco2_Wang"
    n = int(cmd.get("n", 5))
    try:
        data = cache.get_admet_dataset(subset)
        df = data.get_data().head(n)
        records = df.to_dict(orient="records")
        # Convert any numpy scalars to Python types
        for r in records:
            for k, v in r.items():
                try:
                    r[k] = v.item() if hasattr(v, "item") else v
                except Exception:
                    r[k] = str(v)
        return _ok(cmd["id"], subset=subset, sample=records, total_rows=int(len(data.get_data())))
    except Exception as exc:
        return _err(cmd["id"], f"{type(exc).__name__}: {exc}")


def handle_mol_generation_sample(cmd: dict, cache: TDCCache) -> dict:
    """Return example drug SMILES from a generation benchmark dataset."""
    n = int(cmd.get("n", 5))
    try:
        from tdc.generation import MolGen  # type: ignore
        data = MolGen(name="ZINC")
        df = data.get_data().head(n)
        samples = df["smiles"].tolist() if "smiles" in df.columns else df.iloc[:, 0].tolist()
        return _ok(cmd["id"], samples=[str(s) for s in samples])
    except Exception as exc:
        return _err(cmd["id"], f"{type(exc).__name__}: {exc}")


HANDLERS = {
    "ping":                  handle_ping,
    "admet_predict":         handle_admet_predict,
    "load_dataset_sample":   handle_load_dataset_sample,
    "mol_generation_sample": handle_mol_generation_sample,
}


# --------------------------------------------------------- main loop


def main() -> None:
    cache = TDCCache()
    # Announce readiness so the adapter knows cold-start is done
    _write({"id": "_startup", "ok": True, "ready": True})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError as exc:
            _write(_err("_unknown", f"bad JSON: {exc}"))
            continue

        cmd_id = cmd.get("id", "_unspecified")
        name = cmd.get("cmd", "")

        if name == "shutdown":
            _write(_ok(cmd_id))
            return

        handler = HANDLERS.get(name)
        if handler is None:
            _write(_err(cmd_id, f"unknown command: {name}"))
            continue

        try:
            resp = handler(cmd, cache)
        except Exception as exc:
            resp = _err(cmd_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:500]}")
        _write(resp)


if __name__ == "__main__":
    main()
