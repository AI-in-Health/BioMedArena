"""Chemistry TOOL_SPECS — stateless single-shot RDKit/datamol wrappers.

Each function is async-safe (pure computation, no network) and returns a
JSON-serialisable dict. Registered into FunctionCallingRunner's TOOL_SPECS
so the LLM can call them mid-reasoning.

Design: keep each function <100 LOC, no mutation of shared state.
"""

from __future__ import annotations

from typing import Any


# ======================================================================
# Sync worker functions (RDKit is fully synchronous — no asyncio needed)
# ======================================================================


def _canonicalise_sync(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _descriptors_sync(smiles: str) -> dict[str, Any]:
    """Compute common descriptors + Lipinski violations for a canonical SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rb   = Lipinski.NumRotatableBonds(mol)

    # Lipinski rule-of-5 violations
    violations = 0
    if mw > 500:   violations += 1
    if logp > 5:   violations += 1
    if hbd > 5:    violations += 1
    if hba > 10:   violations += 1

    return {
        "MW": round(mw, 3),
        "LogP": round(logp, 3),
        "HBD": hbd,
        "HBA": hba,
        "TPSA": round(tpsa, 3),
        "rotatable_bonds": rb,
        "lipinski_violations": violations,
        "lipinski_pass": violations == 0,
    }


def _fingerprint_sync(smiles: str, fp_type: str = "morgan", radius: int = 2,
                       n_bits: int = 2048) -> list[int]:
    """Return a bit vector (list[int]) for the given fingerprint type."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if fp_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "rdkit":
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    else:
        raise ValueError(f"Unknown fp_type: {fp_type}")

    return list(fp)


def _similarity_sync(smi_a: str, smi_b: str, fp_type: str = "morgan") -> float:
    """Tanimoto similarity between two molecules."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys

    ma = Chem.MolFromSmiles(smi_a)
    mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        raise ValueError("Invalid SMILES")

    if fp_type == "morgan":
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    elif fp_type == "maccs":
        fa = MACCSkeys.GenMACCSKeys(ma)
        fb = MACCSkeys.GenMACCSKeys(mb)
    else:
        raise ValueError(f"Unknown fp_type: {fp_type}")
    return round(DataStructs.TanimotoSimilarity(fa, fb), 4)


def _substructure_match_sync(query_smarts: str, target_smiles: str) -> dict[str, Any]:
    """Check substructure match and return matched atom indices."""
    from rdkit import Chem
    target = Chem.MolFromSmiles(target_smiles)
    query = Chem.MolFromSmarts(query_smarts)
    if target is None:
        raise ValueError(f"Invalid target SMILES: {target_smiles}")
    if query is None:
        raise ValueError(f"Invalid query SMARTS: {query_smarts}")

    matches = target.GetSubstructMatches(query)
    return {
        "match": len(matches) > 0,
        "num_matches": len(matches),
        "atom_indices": [list(m) for m in matches[:10]],  # cap at 10
    }


# ======================================================================
# TOOL_SPECS — registered by FunctionCallingRunner
# ======================================================================

CHEMISTRY_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "mol_from_smiles",
            "description": (
                "Validate a SMILES string and return its canonical form. Use to "
                "normalise user input or verify molecule parseability before other "
                "chemistry tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mol_descriptors",
            "description": (
                "Compute standard molecular descriptors for a SMILES string: "
                "molecular weight (MW), LogP, H-bond donors/acceptors (HBD/HBA), "
                "topological polar surface area (TPSA), rotatable bond count, and "
                "Lipinski rule-of-5 violation count. Use when assessing drug-likeness "
                "or filtering compound libraries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mol_fingerprint",
            "description": (
                "Compute a molecular fingerprint bit vector. Supports Morgan (ECFP-like), "
                "MACCS, and RDKit. Use when comparing molecules or building ML features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "fp_type": {
                        "type": "string",
                        "enum": ["morgan", "maccs", "rdkit"],
                        "default": "morgan",
                    },
                    "radius": {"type": "integer", "default": 2,
                                "description": "Morgan radius (ignored for other fp_type)"},
                    "n_bits": {"type": "integer", "default": 2048,
                                "description": "Fingerprint bit length"},
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mol_similarity",
            "description": (
                "Compute Tanimoto similarity between two molecules using the specified "
                "fingerprint type. Returns a float in [0, 1]. Use when clustering or "
                "retrieving similar compounds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles_a": {"type": "string"},
                    "smiles_b": {"type": "string"},
                    "fp_type": {
                        "type": "string",
                        "enum": ["morgan", "maccs"],
                        "default": "morgan",
                    },
                },
                "required": ["smiles_a", "smiles_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mol_substructure_match",
            "description": (
                "Search a molecule for a substructure defined by a SMARTS query. "
                "Returns whether the query is present and, if so, the matched atom "
                "indices. Useful for toxicophore / pharmacophore detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smarts": {"type": "string"},
                    "target_smiles": {"type": "string"},
                },
                "required": ["query_smarts", "target_smiles"],
            },
        },
    },
]


# ======================================================================
# Handler dispatch — called by FunctionCallingRunner._call
# ======================================================================


def handle_mol_tool(name: str, args: dict[str, Any]) -> str:
    """Synchronous dispatcher for the 5 chemistry tools; returns text string."""
    try:
        if name == "mol_from_smiles":
            c = _canonicalise_sync(args["smiles"])
            return f"canonical={c}" if c else "invalid SMILES"
        if name == "mol_descriptors":
            d = _descriptors_sync(args["smiles"])
            return (
                f"MW={d['MW']} LogP={d['LogP']} HBD={d['HBD']} HBA={d['HBA']} "
                f"TPSA={d['TPSA']} rotatable={d['rotatable_bonds']} "
                f"Lipinski_violations={d['lipinski_violations']} "
                f"lipinski_pass={d['lipinski_pass']}"
            )
        if name == "mol_fingerprint":
            bits = _fingerprint_sync(
                args["smiles"],
                fp_type=args.get("fp_type", "morgan"),
                radius=int(args.get("radius", 2)),
                n_bits=int(args.get("n_bits", 2048)),
            )
            n_set = sum(bits)
            return f"fp_type={args.get('fp_type', 'morgan')} length={len(bits)} bits_set={n_set}"
        if name == "mol_similarity":
            sim = _similarity_sync(
                args["smiles_a"], args["smiles_b"],
                fp_type=args.get("fp_type", "morgan"),
            )
            return f"tanimoto={sim}"
        if name == "mol_substructure_match":
            r = _substructure_match_sync(
                args["query_smarts"], args["target_smiles"]
            )
            return (
                f"match={r['match']} num_matches={r['num_matches']} "
                f"atoms={r['atom_indices']}"
            )
        return f"[unknown chemistry tool: {name}]"
    except Exception as exc:
        return f"[{name} error: {exc}]"


CHEMISTRY_TOOL_NAMES = {
    "mol_from_smiles", "mol_descriptors", "mol_fingerprint",
    "mol_similarity", "mol_substructure_match",
}


# ======================================================================
# admet-ai native predictors
#
# Kept in a second TOOL_SPECS list because they require the optional
# `admet_ai` dep. FunctionCallingRunner merges both lists if the import
# succeeds.
# ======================================================================


ADMET_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "admet_predict_native",
            "description": (
                "Predict ADMET properties for a SMILES using ADMET-AI's "
                "ensemble of chemprop models trained on the Therapeutics "
                "Data Commons. Returns physicochemical properties "
                "(MW, logP, TPSA, QED, Lipinski) plus ~40 ADMET endpoints "
                "(absorption, distribution, metabolism, excretion, toxicity). "
                "Much more comprehensive than mol_descriptors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string of the molecule.",
                    },
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "molecular_property_predict",
            "description": (
                "Predict a specific molecular property via ADMET-AI. "
                "Takes a SMILES and a property name; returns the predicted "
                "value. Property names include e.g. 'HIA_Hou' (human "
                "intestinal absorption), 'Caco2_Wang', 'BBB_Martins' "
                "(blood-brain barrier), 'hERG', 'AMES' (mutagenicity), "
                "'ClinTox', 'DILI' (liver injury)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "property_name": {
                        "type": "string",
                        "description": (
                            "ADMET property name (see admet_ai docs). "
                            "If omitted or unknown, returns all properties."
                        ),
                    },
                },
                "required": ["smiles"],
            },
        },
    },
]


ADMET_TOOL_NAMES = {"admet_predict_native", "molecular_property_predict"}


def handle_admet_tool(name: str, args: dict[str, Any]) -> str:
    """Synchronous dispatcher for the 2 native ADMET tools."""
    # Lazy import to avoid loading chemprop / lightning if tool never used
    try:
        from harness.adapters.chemistry_adapter import ChemistryAdapter
    except ImportError as exc:
        return f"[admet adapter missing: {exc}]"
    adapter = ChemistryAdapter()
    if not adapter._admet_ok:
        return "[admet-ai not installed; run: uv pip install admet-ai chemprop]"
    try:
        smiles = args["smiles"]
        if name == "admet_predict_native":
            resp = adapter.admet_predict_native(smiles)
            if not resp.get("ok"):
                return f"[{name} error: {resp.get('error', 'unknown')}]"
            preds = resp["predictions"]
            # Compact top-N summary to avoid context blow-up
            if isinstance(preds, dict):
                items = list(preds.items())[:25]
                return "; ".join(
                    f"{k}={_fmt(v)}" for k, v in items
                )
            return str(preds)[:15000]
        if name == "molecular_property_predict":
            resp = adapter.admet_predict_native(smiles)
            if not resp.get("ok"):
                return f"[{name} error: {resp.get('error', 'unknown')}]"
            preds = resp["predictions"]
            prop = args.get("property_name")
            if prop and isinstance(preds, dict) and prop in preds:
                return f"{prop}={_fmt(preds[prop])}"
            # Fallback: list all keys
            if isinstance(preds, dict):
                keys = list(preds.keys())
                return f"property '{prop}' not found. Available: {', '.join(keys[:30])}"
            return str(preds)[:15000]
        return f"[unknown admet tool: {name}]"
    except Exception as exc:  # noqa: BLE001
        return f"[{name} error: {exc}]"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
