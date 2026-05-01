"""Retrieval-based tool selection.

Rationale: naive agent stacks put 200+ tools into the LLM prompt at
once, which blows through context and degrades decision quality. This
module retrieves only the top-k (up to ~15) tools relevant to the
current query, plus a small set of always-on "core" tools, keeping the
advertised tool count under 25 regardless of how many tools are
integrated overall.

Architecture
------------
- `ToolRetriever.retrieve(query, context, top_k=15)` returns a subset
  of the input `tools` list.
- Scoring = weighted combination of
    0.65 × cosine(OpenAI embedding of query, embedding of tool description)
    0.25 × domain heuristic boost (SMILES → chem tools, "DICOM" →
        imaging tools, gene symbols → genomics tools, etc.)
    0.10 × name-substring match (cheap boost for obvious aliases).
- Core tools (pubmed_search, calculator_eval, python_exec,
  compute_calculator, code_search, gene_lookup, clinvar_lookup) are
  always included regardless of score, so basic-query flows never
  degrade.
- Embeddings are cached to `data/cache/tool_embeddings.pkl` keyed by a
  hash of (model_name, tool_name, description). Re-embeds only when a
  tool's description changes.

Embedder injection
------------------
Default embedder uses OpenAI's text-embedding-3-small via `LLMClient`.
Tests and offline runs can inject any callable
`embed_fn(texts: list[str]) -> np.ndarray[n, d]`; a deterministic
hashing-trigram fallback is bundled for when no OpenAI key is present
so CI doesn't require network.

Offline fallback is *not* as semantically sharp as real embeddings; it
exists for correctness-of-interface, not for benchmark evaluation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Always-on core tools (addendum §3.1.1)
# ---------------------------------------------------------------------------

CORE_TOOL_NAMES: frozenset[str] = frozenset({
    "pubmed_search",
    "calculator_eval",
    "python_exec",
    "compute_calculator",
    "code_search",
    "gene_lookup",
    "clinvar_lookup",
})


# ---------------------------------------------------------------------------
# Domain heuristic (cheap pre-filter before embedding)
# ---------------------------------------------------------------------------

# Mapping: domain label -> (regex hints in query, tool-name substrings to boost)
_DOMAIN_HINTS: dict[str, tuple[list[str], list[str]]] = {
    "chemistry": (
        [r"\bsmiles\b", r"\binchi\b", r"\bmolecule\b", r"\bcompound\b",
         r"\bligand\b", r"\bdrug[- ]like\b", r"\blipinski\b", r"\badmet\b",
         r"\bsubstructure\b", r"\bfingerprint\b", r"\btanimoto\b"],
        ["mol_", "admet_", "molecular_", "tdc_", "chemistry", "lipinski"],
    ),
    "genomics": (
        [r"\bgene\b", r"\bvariant\b", r"\brsid\b", r"\bclinvar\b",
         r"\bomim\b", r"\bmutation\b", r"\bensembl\b", r"\bentrez\b",
         r"\bpathway\b", r"\bucsc\b", r"\brefseq\b", r"\buniprot\b",
         r"\bBRCA\d\b", r"\bTP53\b", r"\bEGFR\b"],
        ["gene_", "gget_", "mygene", "clinvar", "omim", "uniprot",
         "mcp_tu_", "mcp_biomcp_"],
    ),
    "drug": (
        [r"\bdrug\b", r"\bindication\b", r"\bdaily.?med\b", r"\brxnorm\b",
         r"\brxnav\b", r"\bdose\b", r"\binteraction\b", r"\bside effect\b",
         r"\badverse\b", r"\bfda\b", r"\bopenfda\b"],
        ["rxnav_", "openfda_", "dailymed_", "drug_", "mcp_biomcp_", "mcp_tu_"],
    ),
    "imaging": (
        [r"\bdicom\b", r"\bpacs\b", r"\bct scan\b", r"\bMRI\b", r"\bX-?ray\b",
         r"\bchest\b.*\bradiograph\b", r"\brofile\b", r"\bseries\b",
         r"\binstance uid\b", r"\bsiuid\b"],
        ["dicom_", "mcp_dicom_", "xray", "radiograph"],
    ),
    "clinical": (
        [r"\bcalculator\b", r"\bcharlson\b", r"\bCHA2DS2\b", r"\bsofa\b",
         r"\bapache\b", r"\bicd\b", r"\bCPT\b", r"\bloinc\b", r"\bsnomed\b",
         r"\bfhir\b", r"\bEHR\b"],
        ["calculator", "compute_calculator", "phenoage", "medline"],
    ),
    "protein": (
        [r"\bprotein\b", r"\besm\b", r"\balphafold\b", r"\bstructure\b",
         r"\bpdb\b", r"\bsequence\b", r"\bfold\b", r"\bembed"],
        ["protein_", "alphafold", "mcp_tu_"],
    ),
    "literature": (
        [r"\bpubmed\b", r"\bpaper\b", r"\barticle\b", r"\bliterature\b",
         r"\bstudy\b", r"\bmeta[- ]analysis\b", r"\breview\b", r"\bclinical trial\b"],
        ["pubmed_", "article_", "mcp_biomcp_article", "trial_"],
    ),
}


def score_query_domain(query: str) -> dict[str, float]:
    """Return a {domain: weight_in_0_to_1} dict for this query.

    Multiple domains can fire (e.g. "DICOM study of a lung cancer patient
    with EGFR mutation" → imaging + genomics). Weights are normalised to
    sum to 1 across matching domains. No match → empty dict, caller
    should treat as "no heuristic signal".
    """
    q = query or ""
    hits: Counter[str] = Counter()
    for domain, (patterns, _) in _DOMAIN_HINTS.items():
        for pat in patterns:
            if re.search(pat, q, flags=re.IGNORECASE):
                hits[domain] += 1
    total = sum(hits.values())
    if total == 0:
        return {}
    return {d: n / total for d, n in hits.items()}


def _domain_boost_for_tool(tool_name: str, domain_weights: dict[str, float]) -> float:
    """Return a domain-boost score in [0, 1] for one tool name."""
    if not domain_weights:
        return 0.0
    total = 0.0
    for domain, weight in domain_weights.items():
        _, substrings = _DOMAIN_HINTS[domain]
        if any(sub in tool_name for sub in substrings):
            total += weight
    return min(total, 1.0)


# ---------------------------------------------------------------------------
# Embedder — OpenAI default + deterministic offline fallback
# ---------------------------------------------------------------------------


def _hashing_trigram_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Deterministic offline embedder — hashing trick over character
    trigrams. Not as good as OpenAI; used when no API key available or
    for tests. Returns L2-normalised float32 [n, dim].
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = (t or "").lower()
        # Strip non-alphanumerics to collapse punctuation
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        # Char trigrams across word boundaries
        grams: list[str] = []
        for tok in s.split():
            padded = f"  {tok} "
            grams.extend(padded[k:k+3] for k in range(len(padded) - 2))
        if not grams:
            out[i, 0] = 1.0
            continue
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little")
            sign = 1.0 if (h >> 31) & 1 else -1.0
            out[i, h % dim] += sign
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def _openai_embed(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Synchronous OpenAI embedding call. Raises on failure — caller
    decides whether to fall back.
    """
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # Normalise so we can use dot product as cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------------------------------------------------------------------
# Tool retriever
# ---------------------------------------------------------------------------


@dataclass
class _ToolEntry:
    name: str
    description: str
    spec: dict[str, Any]
    hash: str


_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
_CACHE_FILE = _CACHE_ROOT / "tool_embeddings.pkl"


class ToolRetriever:
    """Pick the top-k tools for a query from a pool of OpenAI-style specs.

    Parameters
    ----------
    tools
        List of OpenAI function-calling specs (same format as
        `harness.eval.function_calling_runner.TOOL_SPECS`).
    embedding_model
        OpenAI embedding model id. `text-embedding-3-small` is a good
        speed/quality tradeoff (default).
    embed_fn
        Optional callable `list[str] -> np.ndarray[n, d]`. Injectable for
        tests / offline runs. If None, try OpenAI; if that fails (missing
        key, network error), fall back to hashing-trigrams.
    cache_path
        Pickle file for the {hash: vector} cache. `None` disables cache.
    core_tools
        Tool names that must always be included. Defaults to the 7
        addendum §3.1.1 core tools.
    """

    def __init__(self,
                   tools: list[dict[str, Any]],
                   embedding_model: str = "text-embedding-3-small",
                   embed_fn: Callable[[list[str]], np.ndarray] | None = None,
                   cache_path: Path | None = _CACHE_FILE,
                   core_tools: frozenset[str] = CORE_TOOL_NAMES):
        self.embedding_model = embedding_model
        self.core_tools = core_tools
        self.cache_path = cache_path
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

        self._entries: list[_ToolEntry] = []
        for spec in tools:
            fn = spec.get("function") or {}
            name = fn.get("name") or ""
            desc = fn.get("description") or ""
            body = f"name={name}\n{desc}"
            h = hashlib.blake2b(
                f"{embedding_model}|{body}".encode(), digest_size=12,
            ).hexdigest()
            self._entries.append(_ToolEntry(name, desc, spec, h))

        self._embeddings: np.ndarray | None = None
        self._embed_fn = embed_fn  # may be None → decided at first call
        self._fallback_used = False

    # ------------------------- cache -----------------------------------

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool_embeddings.pkl unreadable (%s) — ignoring", exc)
            self._cache = {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to write %s: %s", self.cache_path, exc)

    # ------------------------- embedding --------------------------------

    def _resolve_embedder(self) -> Callable[[list[str]], np.ndarray]:
        if self._embed_fn is not None:
            return self._embed_fn
        # No key → immediately go offline (no API noise in tests)
        if not os.environ.get("OPENAI_API_KEY"):
            self._fallback_used = True
            logger.info("OPENAI_API_KEY not set — using offline hashing embedder")
            return _hashing_trigram_embed
        model = self.embedding_model

        def _wrapped(texts: list[str]) -> np.ndarray:
            try:
                return _openai_embed(texts, model=model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI embedding failed (%s) — falling back", exc)
                self._fallback_used = True
                return _hashing_trigram_embed(texts)

        return _wrapped

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return
        # Split into cache-hits and misses
        miss_idx: list[int] = []
        miss_text: list[str] = []
        vecs: list[np.ndarray | None] = [None] * len(self._entries)
        for i, e in enumerate(self._entries):
            cached = self._cache.get(e.hash)
            if cached is not None:
                vecs[i] = cached
            else:
                miss_idx.append(i)
                miss_text.append(f"name={e.name}\n{e.description}")
        if miss_text:
            embed_fn = self._resolve_embedder()
            t0 = time.monotonic()
            new_vecs = embed_fn(miss_text)
            dt = time.monotonic() - t0
            logger.info(
                "Embedded %d tool descriptions in %.2fs (cache hits: %d)",
                len(miss_text), dt, len(self._entries) - len(miss_text),
            )
            for j, i in enumerate(miss_idx):
                v = new_vecs[j]
                vecs[i] = v
                self._cache[self._entries[i].hash] = v
            self._save_cache()
        # Stack — if some cached embeddings have a different dim (e.g. a
        # previous run used hashing, current uses OpenAI), discard the
        # dimension-mismatched ones and re-embed.
        if vecs:
            dims = {v.shape[-1] for v in vecs if v is not None}
            if len(dims) > 1:
                logger.info("Mixed embedding dimensions %s — re-embedding all", dims)
                embed_fn = self._resolve_embedder()
                all_text = [f"name={e.name}\n{e.description}" for e in self._entries]
                stacked = embed_fn(all_text)
                for i, e in enumerate(self._entries):
                    self._cache[e.hash] = stacked[i]
                self._embeddings = stacked
                self._save_cache()
                return
        self._embeddings = np.vstack(vecs) if vecs else np.zeros((0, 1))

    # ------------------------- retrieval --------------------------------

    def retrieve(self, query: str,
                   context: dict[str, Any] | None = None,
                   top_k: int = 15) -> list[dict[str, Any]]:
        """Return top-k tool specs for a query, plus core tools.

        Output order:
          1. Core tools (in their registration order) — always present if
             they exist in the pool.
          2. Top-k non-core tools by score, descending. Total size is at
             most (|core ∩ pool|) + k.
        """
        if not self._entries:
            return []
        self._ensure_embeddings()
        assert self._embeddings is not None

        embed_fn = self._resolve_embedder()
        q_vec = embed_fn([query or ""])[0]
        # Ensure dimension match — if cached embeddings are 256-dim
        # hashing but q_vec is 1536-dim openai, re-project via fallback
        if q_vec.shape[-1] != self._embeddings.shape[-1]:
            q_vec = _hashing_trigram_embed([query or ""])[0]
            if q_vec.shape[-1] != self._embeddings.shape[-1]:
                # Fall back to uniform scores
                sims = np.zeros(len(self._entries), dtype=np.float32)
            else:
                sims = self._embeddings @ q_vec
        else:
            sims = self._embeddings @ q_vec

        domain_weights = score_query_domain(query)
        boosts = np.array(
            [_domain_boost_for_tool(e.name, domain_weights) for e in self._entries],
            dtype=np.float32,
        )
        # Tiny name-substring bonus — helps exact matches like
        # "run admet_predict_native" → tool of the same name.
        q_low = (query or "").lower()
        name_match = np.array(
            [1.0 if e.name.lower() in q_low else 0.0 for e in self._entries],
            dtype=np.float32,
        )

        scores = 0.65 * sims + 0.25 * boosts + 0.10 * name_match

        # Split core vs non-core
        core_picked: list[int] = []
        non_core: list[tuple[float, int]] = []
        for i, e in enumerate(self._entries):
            if e.name in self.core_tools:
                core_picked.append(i)
            else:
                non_core.append((float(scores[i]), i))
        non_core.sort(reverse=True)
        top = [idx for _, idx in non_core[:max(0, top_k)]]
        # Preserve order: core (registration order) then top-k
        ordered_idx = core_picked + top
        # Dedupe while preserving order (just in case)
        seen = set()
        final = []
        for i in ordered_idx:
            if i not in seen:
                seen.add(i)
                final.append(self._entries[i].spec)
        return final

    def last_scores(self, query: str) -> list[tuple[str, float]]:
        """Diagnostic helper — return [(name, combined_score), …] for the
        whole pool, sorted descending. Used for ablation logging."""
        self._ensure_embeddings()
        assert self._embeddings is not None
        embed_fn = self._resolve_embedder()
        q_vec = embed_fn([query or ""])[0]
        if q_vec.shape[-1] != self._embeddings.shape[-1]:
            q_vec = _hashing_trigram_embed([query or ""])[0]
        sims = self._embeddings @ q_vec if q_vec.shape[-1] == self._embeddings.shape[-1] else np.zeros(len(self._entries))
        domain_weights = score_query_domain(query)
        boosts = np.array(
            [_domain_boost_for_tool(e.name, domain_weights) for e in self._entries],
            dtype=np.float32,
        )
        q_low = (query or "").lower()
        name_match = np.array(
            [1.0 if e.name.lower() in q_low else 0.0 for e in self._entries],
            dtype=np.float32,
        )
        scores = 0.65 * np.asarray(sims) + 0.25 * boosts + 0.10 * name_match
        return sorted(
            [(e.name, float(scores[i])) for i, e in enumerate(self._entries)],
            key=lambda x: x[1], reverse=True,
        )

    @property
    def fallback_used(self) -> bool:
        """True if the hashing-trigram fallback produced any embeddings
        on the most recent ensure/retrieve call."""
        return self._fallback_used
