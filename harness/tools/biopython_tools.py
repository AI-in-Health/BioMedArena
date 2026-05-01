"""Biopython TOOL_SPECS — stateless genomics file I/O + sequence ops.

Covers the most common Biopython usage patterns: FASTA / GenBank /
PDB parsing, sequence transforms (reverse complement, translate),
and remote Entrez lookups. Heavy/stateful workflows (alignment trees,
BLAST) stay in the Biopython adapter if/when we add one.
"""

from __future__ import annotations

from typing import Any


BIOPYTHON_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fasta_parse",
            "description": (
                "Parse a FASTA-format string and return a list of "
                "{id, description, sequence, length} records. Good for "
                "inspecting small multi-record FASTA inputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fasta_text": {"type": "string"},
                    "max_records": {"type": "integer", "default": 20},
                },
                "required": ["fasta_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dna_reverse_complement",
            "description": "Return the reverse-complement of a DNA sequence.",
            "parameters": {
                "type": "object",
                "properties": {"sequence": {"type": "string"}},
                "required": ["sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dna_translate",
            "description": (
                "Translate a DNA sequence to protein using the standard "
                "genetic code. Starts at frame 0 unless `frame` is set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "string"},
                    "frame": {"type": "integer", "default": 0,
                                "description": "0, 1 or 2."},
                    "to_stop": {"type": "boolean", "default": True},
                },
                "required": ["sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "genbank_parse",
            "description": (
                "Parse a GenBank-format string and return "
                "{accession, organism, sequence_length, n_features, "
                "features[:20]}."
            ),
            "parameters": {
                "type": "object",
                "properties": {"genbank_text": {"type": "string"}},
                "required": ["genbank_text"],
            },
        },
    },
]


BIOPYTHON_TOOL_NAMES = {
    "fasta_parse", "dna_reverse_complement", "dna_translate", "genbank_parse",
}


# --- sync workers ------------------------------------------------------


def _fasta_parse_sync(text: str, max_records: int = 20) -> list[dict[str, Any]]:
    from io import StringIO
    from Bio import SeqIO
    out = []
    for rec in SeqIO.parse(StringIO(text), "fasta"):
        out.append({
            "id": rec.id,
            "description": rec.description,
            "sequence": str(rec.seq)[:20000],
            "length": len(rec.seq),
        })
        if len(out) >= max_records:
            break
    return out


def _reverse_complement_sync(seq: str) -> str:
    from Bio.Seq import Seq
    return str(Seq(seq).reverse_complement())


def _translate_sync(seq: str, frame: int = 0, to_stop: bool = True) -> str:
    from Bio.Seq import Seq
    s = seq[max(0, min(frame, 2)):]
    # Trim trailing bases so the length is a multiple of 3
    s = s[: len(s) - len(s) % 3]
    return str(Seq(s).translate(to_stop=to_stop))


def _genbank_parse_sync(text: str) -> dict[str, Any]:
    from io import StringIO
    from Bio import SeqIO
    rec = next(SeqIO.parse(StringIO(text), "genbank"))
    features = []
    for f in rec.features[:20]:
        features.append({
            "type": f.type,
            "location": str(f.location),
            "qualifiers": {k: v[:1] for k, v in (f.qualifiers or {}).items()},
        })
    return {
        "accession": rec.id,
        "organism": (rec.annotations or {}).get("organism", ""),
        "sequence_length": len(rec.seq),
        "n_features": len(rec.features),
        "features": features,
    }


def handle_biopython_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "fasta_parse":
            recs = _fasta_parse_sync(
                args["fasta_text"], max_records=int(args.get("max_records", 20)),
            )
            return f"{len(recs)} records; first: {recs[0] if recs else 'none'}"
        if name == "dna_reverse_complement":
            return _reverse_complement_sync(args["sequence"])
        if name == "dna_translate":
            return _translate_sync(
                args["sequence"],
                frame=int(args.get("frame", 0)),
                to_stop=bool(args.get("to_stop", True)),
            )
        if name == "genbank_parse":
            r = _genbank_parse_sync(args["genbank_text"])
            return (
                f"accession={r['accession']} organism={r['organism']} "
                f"length={r['sequence_length']} features={r['n_features']}"
            )
        return f"[unknown biopython tool: {name}]"
    except Exception as exc:  # noqa: BLE001
        return f"[{name} error: {exc}]"
