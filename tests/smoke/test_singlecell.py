"""Smoke tests for scanpy + anndata.

Generates a synthetic AnnData (100 cells x 50 genes) in tmp dir and runs
the adapter's qc / normalise / umap pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("scanpy")
pytest.importorskip("anndata")


@pytest.fixture
def synthetic_h5ad(tmp_path: Path) -> Path:
    import anndata as ad
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(42)
    n_obs, n_vars = 80, 40
    X = rng.poisson(5, size=(n_obs, n_vars)).astype("float32")
    # Inject 2 MT genes for QC path
    var_names = [f"Gene{i}" for i in range(n_vars - 2)] + ["MT-ND1", "MT-CO1"]
    obs_names = [f"Cell{i}" for i in range(n_obs)]
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"leiden": ["0"] * (n_obs // 2) + ["1"] * (n_obs - n_obs // 2)},
                          index=obs_names),
        var=pd.DataFrame(index=var_names),
    )
    out = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(str(out))
    return out


def test_singlecell_adapter_in_registry():
    from harness.adapters import ADAPTER_REGISTRY
    assert "SingleCellAdapter" in ADAPTER_REGISTRY


@pytest.mark.asyncio
async def test_qc_default(synthetic_h5ad):
    from harness.adapters.singlecell_adapter import SingleCellAdapter
    a = SingleCellAdapter()
    assert a.available
    r = await a.run("QC this file", context={"h5ad_path": str(synthetic_h5ad)})
    raw = r["raw"]
    assert raw["n_obs"] == 80
    assert raw["n_vars"] == 40
    assert raw["mt_genes_found"] == 2
    assert r["confidence"] > 0.8


@pytest.mark.asyncio
async def test_normalise_and_umap(synthetic_h5ad):
    from harness.adapters.singlecell_adapter import SingleCellAdapter
    a = SingleCellAdapter()
    r1 = await a.run("normalise", context={
        "h5ad_path": str(synthetic_h5ad), "operation": "normalise",
    })
    assert r1["raw"]["ok"] is True

    r2 = await a.run("umap", context={
        "h5ad_path": str(synthetic_h5ad), "operation": "neighbors_umap",
    })
    assert r2["raw"]["umap_shape"] == [80, 2]


@pytest.mark.asyncio
async def test_rank_genes_groups(synthetic_h5ad):
    from harness.adapters.singlecell_adapter import SingleCellAdapter
    a = SingleCellAdapter()
    # Must first normalise then compute markers
    await a.run("normalise", context={
        "h5ad_path": str(synthetic_h5ad), "operation": "normalise",
    })
    r = await a.run("markers", context={
        "h5ad_path": str(synthetic_h5ad),
        "operation": "rank_genes_groups",
        "groupby": "leiden",
    })
    raw = r["raw"]
    assert raw["groupby"] == "leiden"
    assert raw["n_groups"] == 2


@pytest.mark.asyncio
async def test_missing_path():
    from harness.adapters.singlecell_adapter import SingleCellAdapter
    a = SingleCellAdapter()
    r = await a.run("no path", context={})
    assert r["confidence"] < 0.2


@pytest.mark.asyncio
async def test_unknown_operation(synthetic_h5ad):
    from harness.adapters.singlecell_adapter import SingleCellAdapter
    a = SingleCellAdapter()
    r = await a.run("bad op", context={
        "h5ad_path": str(synthetic_h5ad), "operation": "nonsense_op",
    })
    assert "unknown" in r["answer"].lower()
