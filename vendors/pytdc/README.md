# PyTDC vendor — isolated subprocess worker

This directory hosts a small JSON-over-stdio server (`entry.py`) running in
its own venv so that PyTDC's older numpy / pandas / datasets / rdkit pins
don't downgrade the main `.venv311`.

## First-time setup

```bash
# From the repo root:
uv venv --python 3.11 vendors/pytdc/.venv311
uv pip install --python vendors/pytdc/.venv311/bin/python \
    -r vendors/pytdc/requirements.txt
```

The venv is gitignored. Only `entry.py`, `README.md`, and `requirements.txt`
are committed.

## Usage

Do NOT invoke `entry.py` directly. Go through the adapter:

```python
from harness.adapters.pytdc_adapter import PyTDCAdapter

adapter = PyTDCAdapter()
resp = await adapter.admet_predict(
    "CC(=O)Oc1ccccc1C(=O)O",
    endpoints=["Caco2_Wang", "Lipophilicity_AstraZeneca"],
)
```

The adapter keeps one subprocess alive per Python process (cold start
~10-30s, subsequent calls <1s).

## Protocol

Line-delimited JSON on stdin → line-delimited JSON on stdout.

```
→ {"id": "abc", "cmd": "admet_predict", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "endpoints": ["Caco2_Wang"]}
← {"id": "abc", "ok": true, "predictions": {"Caco2_Wang": -4.31}, ...}
```

Supported commands: `ping`, `admet_predict`, `load_dataset_sample`,
`mol_generation_sample`, `shutdown`.

## Why a subprocess?

PyTDC pins older versions of ~9 packages that would downgrade the main venv
(numpy, pandas, datasets, rdkit, pyarrow, huggingface-hub, fsspec, multiprocess,
dill). Isolation keeps the main venv stable while still giving us TDC's
datasets and generation benchmarks.
