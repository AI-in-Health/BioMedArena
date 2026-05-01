"""Unified retrieval client registry.

All clients expose `async def summary(query: str) -> dict[str, Any]` with keys:
{summary, evidence, ...}. Graceful degradation: empty summary on failure or no key.
"""

from __future__ import annotations

from harness.adapters.retrieval.rxnav_client import RxNavClient
from harness.adapters.retrieval.openfda_client import OpenFDAClient
from harness.adapters.retrieval.dailymed_client import DailyMedClient
from harness.adapters.retrieval.omim_client import OMIMClient
from harness.adapters.retrieval.orphanet_client import OrphanetClient
from harness.adapters.retrieval.medlineplus_client import MedlinePlusClient

# NCBI adapter is separate (pre-existing)
from harness.adapters.ncbi_tools_adapter import NCBIToolsAdapter


RETRIEVAL_CLIENTS = {
    "ncbi": NCBIToolsAdapter,
    "rxnav": RxNavClient,
    "openfda": OpenFDAClient,
    "dailymed": DailyMedClient,
    "omim": OMIMClient,
    "orphanet": OrphanetClient,
    "medlineplus": MedlinePlusClient,
}

__all__ = [
    "RETRIEVAL_CLIENTS",
    "RxNavClient",
    "OpenFDAClient",
    "DailyMedClient",
    "OMIMClient",
    "OrphanetClient",
    "MedlinePlusClient",
    "NCBIToolsAdapter",
]
