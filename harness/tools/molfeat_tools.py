"""Molfeat TOOL_SPECS — richer molecular featurizers beyond RDKit.

molfeat bundles dozens of featurizer types (Morgan, MACCS, physchem,
graph-based, pretrained transformer fingerprints). These TOOL_SPECS
expose the lightweight ones; heavy pretrained featurizers stay off
the LLM surface to avoid unexpected model downloads on first call.
"""

from __future__ import annotations

from typing import Any


MOLFEAT_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "molfeat_featurize",
            "description": (
                "Compute a molecular feature vector for a SMILES using "
                "molfeat. `featurizer` is one of: 'ecfp' (Morgan 2048), "
                "'maccs' (166-bit), 'mordred' (descriptors, slower), "
                "'rdkit' (default physchem). Returns summary stats of "
                "the vector (first 10 dims, mean, nnz); full vectors "
                "are truncated to avoid context blow-up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "featurizer": {
                        "type": "string",
                        "enum": ["ecfp", "maccs", "mordred", "rdkit"],
                        "default": "ecfp",
                    },
                },
                "required": ["smiles"],
            },
        },
    },
]


MOLFEAT_TOOL_NAMES = {"molfeat_featurize"}


def _featurize_sync(smiles: str, featurizer: str = "ecfp") -> dict[str, Any]:
    from molfeat.calc import FPCalculator
    import numpy as np
    fp_map = {"ecfp": "ecfp", "maccs": "maccs", "rdkit": "desc3D"}
    if featurizer == "mordred":
        from molfeat.calc import MordredDescriptors
        calc = MordredDescriptors(replace_nan=True)
    else:
        calc = FPCalculator(fp_map.get(featurizer, "ecfp"))
    vec = np.asarray(calc(smiles), dtype=np.float32)
    return {
        "featurizer": featurizer,
        "dim": int(vec.shape[0]),
        "first10": vec[:10].tolist(),
        "mean": float(vec.mean()),
        "nnz": int((vec != 0).sum()),
    }


def handle_molfeat_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "molfeat_featurize":
            r = _featurize_sync(
                args["smiles"], featurizer=args.get("featurizer", "ecfp"),
            )
            return (
                f"featurizer={r['featurizer']} dim={r['dim']} "
                f"mean={r['mean']:.4f} nnz={r['nnz']} "
                f"first10={[round(v, 3) for v in r['first10']]}"
            )
        return f"[unknown molfeat tool: {name}]"
    except Exception as exc:  # noqa: BLE001
        return f"[{name} error: {exc}]"
