"""Single-cell adapter — scanpy + anndata stateful analysis.

Routing: adapter (stateful — caches AnnData objects between calls so a
subsequent `umap` op doesn't reload the whole h5ad).

Operations via context['operation']:
    qc             (default) — n_obs, n_vars, pct_counts_mt stats
    normalise                  — normalise_total + log1p (idempotent in cache)
    neighbors_umap             — neighbors + UMAP + coords
    rank_genes_groups          — Wilcoxon markers; requires a .obs column key
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase


# Silence scanpy's banner on import
os.environ.setdefault("SCANPY_VERBOSITY", "warning")


class SingleCellAdapter(AdapterBase):
    name = "singlecell"
    modality = "wearable"  # closest existing modality bucket; ok for routing
    description = (
        "Single-cell RNA-seq analysis via scanpy + anndata. Operations: "
        "qc (default), normalise, neighbors_umap, rank_genes_groups. "
        "Requires context['h5ad_path'] or context['adata_id'] for a cached object."
    )

    # Module-level AnnData cache, keyed by (path, mtime)
    _cache: dict[tuple[str, int], Any] = {}

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        try:
            import scanpy  # noqa: F401
            import anndata  # noqa: F401
            self._ok = True
        except ImportError:
            self._ok = False
            self.mark_unavailable("scanpy or anndata not installed")

    def capabilities(self) -> list[str]:
        return [
            "single_cell_qc",
            "single_cell_normalise",
            "single_cell_umap",
            "single_cell_markers",
            "rna_seq_analysis",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._ok:
            return self.result(answer=self.unavailable_reason, confidence=0.0)

        ctx = context or {}
        path = ctx.get("h5ad_path")
        op = ctx.get("operation", "qc")

        if not path:
            return self.result(
                answer="SingleCellAdapter needs context['h5ad_path'] pointing at an .h5ad file.",
                confidence=0.1,
            )
        p = Path(path)
        if not p.exists():
            return self.result(answer=f"h5ad path missing: {path}", confidence=0.0)

        try:
            result = await asyncio.to_thread(self._run_op, str(p), op, ctx)
        except Exception as exc:
            return self.result(answer=f"scanpy error: {type(exc).__name__}: {exc}", confidence=0.0)

        summary = "\n".join(f"{k}: {v}" for k, v in result.items() if not k.startswith("_"))
        return self.result(
            answer=summary,
            evidence=[f"scanpy {op} on {p.name}"],
            confidence=0.9,
            raw=result,
        )

    def _run_op(self, path: str, op: str, ctx: dict[str, Any]) -> dict[str, Any]:
        adata = self._load_cached(path)
        if op == "qc":
            return _op_qc(adata)
        if op == "normalise":
            _op_normalise(adata)
            self._invalidate_cache(path)
            return {"ok": True, "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
                    "after": "normalise_total+log1p"}
        if op == "neighbors_umap":
            return _op_umap(adata)
        if op == "rank_genes_groups":
            groupby = ctx.get("groupby", "leiden")
            return _op_markers(adata, groupby)
        return {"error": f"unknown operation: {op}"}

    @classmethod
    def _load_cached(cls, path: str) -> Any:
        import anndata as ad
        stat = os.stat(path)
        key = (path, int(stat.st_mtime))
        if key not in cls._cache:
            # Evict older entries for the same path
            for k in list(cls._cache.keys()):
                if k[0] == path:
                    del cls._cache[k]
            cls._cache[key] = ad.read_h5ad(path)
        return cls._cache[key]

    @classmethod
    def _invalidate_cache(cls, path: str) -> None:
        for k in list(cls._cache.keys()):
            if k[0] == path:
                del cls._cache[k]


# ======================================================================
# Ops (sync; scanpy is synchronous)
# ======================================================================


def _op_qc(adata) -> dict[str, Any]:
    import numpy as np
    import scanpy as sc

    # Mitochondrial flag (handles both human MT- and mouse mt-)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    # scanpy's default percent_top=[50,100,200,500] crashes when n_vars < 50.
    # Only pass values that fit.
    feasible_top = [t for t in [50, 100, 200, 500] if t < adata.n_vars]
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], inplace=True, log1p=False,
        percent_top=feasible_top or None,
    )

    X = adata.X
    # Sparse -> dense for small summaries; if big, skip dense conversion
    try:
        mean_counts = float(np.asarray(X.mean()))
    except Exception:
        mean_counts = float(0.0)

    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "mean_counts_per_cell": round(mean_counts, 3),
        "pct_counts_mt_mean": round(float(adata.obs.get("pct_counts_mt", [0]).mean()), 3)
            if "pct_counts_mt" in adata.obs.columns else 0.0,
        "mt_genes_found": int(adata.var["mt"].sum()),
    }


def _op_normalise(adata) -> None:
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=10_000)
    sc.pp.log1p(adata)


def _op_umap(adata) -> dict[str, Any]:
    import scanpy as sc
    if "neighbors" not in adata.uns:
        sc.pp.pca(adata, n_comps=min(30, adata.n_vars - 1, adata.n_obs - 1))
        sc.pp.neighbors(adata, n_neighbors=min(15, adata.n_obs - 1))
    sc.tl.umap(adata)
    umap = adata.obsm["X_umap"]
    return {
        "n_obs": int(adata.n_obs),
        "umap_shape": list(umap.shape),
        "umap_x_range": [round(float(umap[:, 0].min()), 3),
                          round(float(umap[:, 0].max()), 3)],
        "umap_y_range": [round(float(umap[:, 1].min()), 3),
                          round(float(umap[:, 1].max()), 3)],
    }


def _op_markers(adata, groupby: str) -> dict[str, Any]:
    import scanpy as sc
    if groupby not in adata.obs.columns:
        return {"error": f"groupby key '{groupby}' not in adata.obs"}
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", n_genes=10)
    names = adata.uns["rank_genes_groups"]["names"]
    groups = list(names.dtype.names)
    top_per_group = {g: list(names[g][:5]) for g in groups}
    return {"groupby": groupby, "n_groups": len(groups), "top_markers": top_per_group}
