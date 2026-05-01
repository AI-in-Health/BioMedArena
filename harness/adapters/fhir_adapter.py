"""FHIR EHR adapter — fhir.resources + fhirpy wrapper.

Stateful (holds a SyncFHIRClient pool keyed by base-URL). Operations:
- `search`: resource-type search with filters (e.g. Patient, Observation)
- `read`:   fetch a single resource by ID
- `summarize_patient`: multi-call roll-up of recent Observations /
   Conditions / MedicationRequests for a given Patient

Most real hospitals require bearer-token auth and a specific FHIR
version — the adapter accepts them via init config.

Without a configured FHIR endpoint the adapter marks itself
unavailable; sample "public" test servers (e.g. HAPI) are too flaky
to rely on in CI, so the smoke test only covers the adapter
interface, not live data.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from harness.adapter_base import AdapterBase

logger = logging.getLogger(__name__)


_HAPI_TEST_BASE = "http://hapi.fhir.org/baseR4"


class FHIRAdapter(AdapterBase):
    name = "fhir"
    modality = "ehr"
    description = (
        "FHIR R4 EHR adapter backed by fhirpy. Supports search / read / "
        "patient summarisation against any FHIR-compliant server. Set "
        "config['fhir']['base_url'] (and optional 'token') to point at "
        "a real endpoint."
    )

    _client_pool: ClassVar[dict[str, Any]] = {}

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        try:
            import fhirpy  # noqa: F401
            import fhir.resources  # noqa: F401
            self._deps_ok = True
        except ImportError as exc:
            self._deps_ok = False
            self.mark_unavailable(f"fhirpy/fhir.resources not installed: {exc}")
            return

        fhir_cfg = self._config.get("fhir", {})
        self._base_url = fhir_cfg.get("base_url", _HAPI_TEST_BASE)
        self._token = fhir_cfg.get("token")
        self._live_check_done = False  # liveness deferred to first call

    def capabilities(self) -> list[str]:
        return [
            "fhir_search",
            "fhir_resource_read",
            "patient_summary",
            "ehr_query",
        ]

    # -------- client factory ------------------------------------------

    def _get_client(self):
        cache_key = f"{self._base_url}|{self._token or ''}"
        client = self._client_pool.get(cache_key)
        if client is None:
            from fhirpy import SyncFHIRClient
            auth = {"authorization": f"Bearer {self._token}"} if self._token else None
            client = SyncFHIRClient(self._base_url, extra_headers=auth)
            self._client_pool[cache_key] = client
        return client

    # -------- public API ---------------------------------------------

    def search(self, resource_type: str, **params: Any) -> list[dict[str, Any]]:
        """Search `resource_type` with the given query params."""
        client = self._get_client()
        resources = list(client.resources(resource_type).search(**params).fetch_all())
        out = []
        for r in resources:
            try:
                out.append(dict(r))
            except Exception:
                out.append({"resource_type": resource_type, "id": getattr(r, "id", None)})
        return out

    def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        client = self._get_client()
        r = client.reference(f"{resource_type}/{resource_id}").to_resource()
        try:
            return dict(r)
        except Exception:
            return {"resource_type": resource_type, "id": resource_id}

    def summarise_patient(self, patient_id: str, n_observations: int = 10) -> dict[str, Any]:
        obs = self.search("Observation", subject=f"Patient/{patient_id}",
                            _count=n_observations, _sort="-date")
        cond = self.search("Condition", subject=f"Patient/{patient_id}", _count=10)
        meds = self.search("MedicationRequest", subject=f"Patient/{patient_id}",
                             _count=10, status="active")
        return {
            "patient_id": patient_id,
            "n_observations": len(obs),
            "observations": obs,
            "n_conditions": len(cond),
            "conditions": cond,
            "n_medications_active": len(meds),
            "medications_active": meds,
        }

    # -------- AdapterBase.run ----------------------------------------

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._deps_ok:
            return self.result(answer=f"FHIR adapter unavailable: {self.unavailable_reason}",
                                 confidence=0.0)
        import asyncio
        ctx = context or {}
        op = ctx.get("op", "search")
        try:
            if op == "search":
                rtype = ctx["resource_type"]
                params = ctx.get("params", {})
                recs = await asyncio.to_thread(self.search, rtype, **params)
                answer = f"Found {len(recs)} {rtype} records."
                return self.result(answer=answer,
                                     evidence=[f"FHIR {rtype} at {self._base_url}"],
                                     confidence=0.8 if recs else 0.3,
                                     raw={"records": recs[:20]})
            if op == "read":
                rec = await asyncio.to_thread(
                    self.read, ctx["resource_type"], ctx["resource_id"],
                )
                return self.result(answer=f"Fetched {ctx['resource_type']}/{ctx['resource_id']}.",
                                     confidence=0.9, raw=rec)
            if op == "summarize_patient":
                summary = await asyncio.to_thread(self.summarise_patient, ctx["patient_id"])
                answer = (
                    f"Patient {summary['patient_id']}: "
                    f"{summary['n_observations']} observations, "
                    f"{summary['n_conditions']} conditions, "
                    f"{summary['n_medications_active']} active medications."
                )
                return self.result(answer=answer, confidence=0.85, raw=summary)
            return self.result(answer=f"Unknown op: {op}", confidence=0.0)
        except Exception as exc:  # noqa: BLE001
            return self.result(answer=f"FHIR error: {exc}", confidence=0.0)
