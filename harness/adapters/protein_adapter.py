"""Protein adapter — ESM embeddings + AlphaFold-DB structure lookup.

Routing: adapter for heavy state (ESM model weights 30MB–10GB loaded lazily);
plus a separate stateless AlphaFold-DB HTTP tool in harness/tools/protein_tools.py.

Operations via context['operation']:
    embed          (default) — ESM-2 residue embeddings
    alphafold_db              — fetch AlphaFold-DB entry for a UniProt ID
    both                      — run both if sequence + uniprot_id supplied

Model size is config-driven. Read from `config['protein']['esm_model_size']`
or `context['esm_model_size']` (per-call override). Default: "8M" (smallest).
See `ESM_MODEL_MAP` below for the 3 supported sizes.

Only the 8M model is pre-warmed implicitly on first call; 650M / 3B load on
demand. For benchmark-critical protein tasks prefer "650M" — it's the
standard ESM-2 size used in most published baselines.
"""

from __future__ import annotations

import asyncio
from typing import Any

from harness.adapter_base import AdapterBase


# Size alias → (fair-esm pretrained function name, final-layer index, embed dim)
# Only 'name' is strictly needed; layer index and dim are for reference and
# for figuring out which layer to return from the model call.
ESM_MODEL_MAP: dict[str, tuple[str, int, int]] = {
    "8M":   ("esm2_t6_8M_UR50D",   6, 320),
    "650M": ("esm2_t33_650M_UR50D", 33, 1280),
    "3B":   ("esm2_t36_3B_UR50D",  36, 2560),
}

_DEFAULT_ESM_SIZE = "8M"


class ProteinAdapter(AdapterBase):
    name = "protein"
    modality = "genomics"  # closest existing bucket
    description = (
        "Protein analysis: ESM-2 residue embeddings + AlphaFold-DB structure "
        "metadata lookup. Operations: embed (default, needs sequence), "
        "alphafold_db (needs uniprot_id), both."
    )

    # Class-level model cache — keyed by size alias so 8M / 650M / 3B
    # can coexist in the same process without re-downloading.
    _esm_cache: dict[str, tuple[Any, Any]] = {}

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        # Resolve default model size: config['protein']['esm_model_size']
        # falls back to _DEFAULT_ESM_SIZE; per-call override is allowed via
        # context['esm_model_size'].
        protein_cfg = self._config.get("protein", {}) if isinstance(self._config, dict) else {}
        self._default_size = str(protein_cfg.get("esm_model_size", _DEFAULT_ESM_SIZE))
        if self._default_size not in ESM_MODEL_MAP:
            # Silent sanity-fix — fall back to 8M if config is typo'd
            self._default_size = _DEFAULT_ESM_SIZE
        try:
            import esm  # noqa: F401
            self._ok = True
        except ImportError:
            self._ok = False
            self.mark_unavailable("fair-esm not installed")

    def capabilities(self) -> list[str]:
        return [
            "protein_embedding",
            "esm2_residue_features",
            "alphafold_db_lookup",
            "protein_structure_metadata",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._ok:
            return self.result(answer=self.unavailable_reason, confidence=0.0)

        ctx = context or {}
        op = ctx.get("operation", "embed")

        size = self._resolve_size(ctx.get("esm_model_size"))

        if op == "embed":
            seq = ctx.get("sequence")
            if not seq:
                return self.result(
                    answer="ProteinAdapter embed needs context['sequence'] (protein sequence).",
                    confidence=0.1,
                )
            try:
                result = await asyncio.to_thread(self._embed_sync, seq, size)
            except Exception as exc:
                return self.result(answer=f"ESM error: {type(exc).__name__}: {exc}", confidence=0.0)
            return self.result(
                answer=f"ESM-2 residue embeddings: shape={result['embedding_shape']} model={result['model']}",
                evidence=[f"ESM model: {result['model']} (size={size})"],
                confidence=0.85,
                raw=result,
            )

        if op == "alphafold_db":
            uid = ctx.get("uniprot_id")
            if not uid:
                return self.result(
                    answer="AlphaFold-DB lookup needs context['uniprot_id'].",
                    confidence=0.1,
                )
            from harness.tools.protein_tools import _alphafold_db_sync
            result = await asyncio.to_thread(_alphafold_db_sync, uid)
            if not result.get("ok"):
                return self.result(answer=f"AlphaFold-DB: {result.get('error', 'unknown')}", confidence=0.1)
            summary = (
                f"UniProt {uid}: pLDDT mean={result['plddt_mean']:.1f}, "
                f"length={result['sequence_length']}, PDB URL: {result.get('pdb_url', '-')}"
            )
            return self.result(answer=summary, evidence=[f"AlphaFold-DB {uid}"], confidence=0.9, raw=result)

        if op == "both":
            # Concurrent embed + lookup
            seq = ctx.get("sequence")
            uid = ctx.get("uniprot_id")
            tasks = []
            if seq:
                tasks.append(("embed", asyncio.to_thread(self._embed_sync, seq, size)))
            if uid:
                from harness.tools.protein_tools import _alphafold_db_sync
                tasks.append(("alphafold_db", asyncio.to_thread(_alphafold_db_sync, uid)))
            if not tasks:
                return self.result(
                    answer="'both' needs at least one of context['sequence'] or context['uniprot_id'].",
                    confidence=0.1,
                )
            results = await asyncio.gather(*(t[1] for t in tasks), return_exceptions=True)
            merged: dict[str, Any] = {}
            for (name, _), res in zip(tasks, results):
                merged[name] = {"error": str(res)} if isinstance(res, Exception) else res
            return self.result(
                answer=f"Protein combined: {list(merged.keys())}",
                confidence=0.8,
                raw=merged,
            )

        return self.result(
            answer=f"Unknown operation: {op}. Supported: embed, alphafold_db, both",
            confidence=0.1,
        )

    # -----------------------------------------------------------
    # ESM sync worker
    # -----------------------------------------------------------

    def _resolve_size(self, override: str | None) -> str:
        """Pick the size for this call — explicit override > adapter default."""
        if override and str(override) in ESM_MODEL_MAP:
            return str(override)
        return self._default_size

    @classmethod
    def _load_esm(cls, size: str) -> tuple[Any, Any, int]:
        """Lazily load + cache the requested ESM size. Returns (model, alphabet, layer_idx)."""
        if size not in ESM_MODEL_MAP:
            raise ValueError(
                f"Unsupported ESM size '{size}'. Available: {sorted(ESM_MODEL_MAP)}"
            )
        model_fn, layer_idx, _dim = ESM_MODEL_MAP[size]
        if size in cls._esm_cache:
            model, alphabet = cls._esm_cache[size]
            return model, alphabet, layer_idx
        import esm
        loader = getattr(esm.pretrained, model_fn)
        model, alphabet = loader()
        model.eval()
        cls._esm_cache[size] = (model, alphabet)
        return model, alphabet, layer_idx

    def _embed_sync(self, sequence: str, size: str) -> dict[str, Any]:
        import torch
        model, alphabet, layer_idx = self._load_esm(size)
        model_fn, _, _ = ESM_MODEL_MAP[size]
        batch_converter = alphabet.get_batch_converter()
        data = [("peptide", sequence)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            out = model(tokens, repr_layers=[layer_idx], return_contacts=False)
        # Final-layer residue representations
        reps = out["representations"][layer_idx]
        # Shape: (1, seq_len+2, embed_dim). Strip start/end tokens.
        usable = reps[0, 1:-1]
        return {
            "model": model_fn,
            "size": size,
            "sequence_length": len(sequence),
            "embedding_shape": list(usable.shape),
            "embedding_mean": round(float(usable.mean()), 4),
            "embedding_std": round(float(usable.std()), 4),
        }
