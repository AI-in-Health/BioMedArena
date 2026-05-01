"""Chemistry adapter — RDKit + datamol wrapper + admet-ai ensemble.

Routing rule (per GAP_REPORT_ADDENDUM): Adapter for stateful/heavy ops,
TOOL_SPECS entries (see `harness/tools/chemistry_tools.py`) for stateless
LLM-callable single-shot functions.

This adapter holds:
- An LRU cache of parsed `Mol` objects keyed by canonical SMILES so
  repeated queries on the same molecule don't re-parse (RDKit).
- A class-level lazy singleton of `admet_ai.ADMETModel` so the 40+
  chemprop ensembles are loaded once and shared across predict calls.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any, ClassVar

from harness.adapter_base import AdapterBase


class ChemistryAdapter(AdapterBase):
    name = "chemistry"
    modality = "drug"
    description = (
        "RDKit + admet-ai chemistry analyser. Computes molecular descriptors, "
        "Lipinski rule-of-5 flags, canonical SMILES, drug-likeness summaries, "
        "and native ADMET predictions (PK / toxicity / physicochemical) via "
        "an ensemble of chemprop models trained on the Therapeutics Data Commons."
    )

    # Class-level admet-ai model cache (shared across instances) + lock.
    _admet_model: ClassVar[Any] = None
    _admet_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        try:
            import rdkit  # noqa: F401
            self._rdkit_ok = True
        except ImportError:
            self._rdkit_ok = False
            self.mark_unavailable("rdkit not installed")
        # admet-ai is optional — fall back gracefully if absent
        try:
            import admet_ai  # noqa: F401
            self._admet_ok = True
        except ImportError:
            self._admet_ok = False

    def capabilities(self) -> list[str]:
        caps = [
            "molecule_analysis",
            "smiles_validation",
            "lipinski_rule",
            "descriptor_calculation",
            "similarity",
            "fingerprinting",
            "substructure_matching",
        ]
        if self._admet_ok:
            caps.extend([
                "admet_prediction_native",
                "molecular_property_prediction",
                "drugbank_approved_percentile",
            ])
        return caps

    # ------------------------------------------------------------------
    # admet-ai native predictor
    # ------------------------------------------------------------------

    @classmethod
    def _get_admet_model(cls) -> Any:
        """Lazy singleton for `admet_ai.ADMETModel`.

        First call incurs ~0.5s model load. Subsequent calls reuse the
        same ensemble. Thread-safe via class-level lock.
        """
        if cls._admet_model is not None:
            return cls._admet_model
        with cls._admet_lock:
            if cls._admet_model is None:
                from admet_ai import ADMETModel
                cls._admet_model = ADMETModel()
        return cls._admet_model

    def admet_predict_native(self, smiles: str | list[str]) -> dict[str, Any]:
        """Run native ADMET-AI prediction on one SMILES (or list).

        Returns a dict:
            {"ok": bool, "predictions": {name: value, ...}}  (single input)
            {"ok": bool, "predictions": [{name: value, ...}, ...]}  (list input)

        Physicochemical keys (molecular_weight, logP, tpsa, QED, Lipinski)
        plus ADMET endpoints (absorption, distribution, metabolism,
        excretion, toxicity). See admet_ai docs for full key list.
        """
        if not self._admet_ok:
            return {"ok": False, "error": "admet-ai not installed"}
        if not self._rdkit_ok:
            return {"ok": False, "error": "rdkit not installed"}
        try:
            model = self._get_admet_model()
            out = model.predict(smiles)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # Normalise pandas DataFrame (batch) → list-of-dicts
        if hasattr(out, "to_dict"):
            rows = out.to_dict(orient="records")
            return {"ok": True, "predictions": rows}
        return {"ok": True, "predictions": out}

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._rdkit_ok:
            return self.result(
                answer="RDKit unavailable — install via `uv pip install rdkit datamol`.",
                confidence=0.0,
            )

        ctx = context or {}
        smiles_list = ctx.get("smiles_list") or (
            [ctx["smiles"]] if ctx.get("smiles") else []
        )
        if not smiles_list:
            return self.result(
                answer=(
                    "ChemistryAdapter needs context['smiles'] (single) or "
                    "context['smiles_list'] (batch)."
                ),
                confidence=0.1,
            )

        import asyncio
        results = await asyncio.to_thread(self._batch_analyse, smiles_list)
        answer_lines = []
        evidence = []
        confidences: list[float] = []
        for sm, r in zip(smiles_list, results):
            if "error" in r:
                answer_lines.append(f"{sm} — error: {r['error']}")
                confidences.append(0.0)
            else:
                ro5 = r["lipinski_violations"]
                flag = "PASS" if ro5 == 0 else f"{ro5} violations"
                answer_lines.append(
                    f"{r['canonical_smiles']}: MW={r['MW']:.1f}, LogP={r['LogP']:.2f}, "
                    f"HBD={r['HBD']}, HBA={r['HBA']}, TPSA={r['TPSA']:.1f}, "
                    f"rotatable={r['rotatable_bonds']}, Lipinski={flag}"
                )
                evidence.append(
                    f"RDKit descriptors for {r['canonical_smiles']}"
                )
                confidences.append(0.95)

        overall_conf = (sum(confidences) / max(len(confidences), 1)) if confidences else 0.0
        return self.result(
            answer="\n".join(answer_lines),
            evidence=evidence,
            confidence=overall_conf,
            raw={"per_molecule": results},
        )

    # Sync worker — runs in thread via asyncio.to_thread
    def _batch_analyse(self, smiles_list: list[str]) -> list[dict[str, Any]]:
        from harness.tools.chemistry_tools import (
            _descriptors_sync,
            _canonicalise_sync,
        )
        out = []
        for sm in smiles_list:
            try:
                canonical = _canonicalise_sync(sm)
                if canonical is None:
                    out.append({"error": "invalid SMILES"})
                    continue
                desc = _descriptors_sync(canonical)
                out.append({"canonical_smiles": canonical, **desc})
            except Exception as exc:
                out.append({"error": str(exc)})
        return out


# ----------------------------------------------------------------------
# LRU cache — module-level so it's shared across adapter instances.
# ----------------------------------------------------------------------
@lru_cache(maxsize=512)
def _cached_mol(smiles: str):
    """Parse a SMILES string to an RDKit Mol (cached)."""
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)
