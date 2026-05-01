#!/usr/bin/env python3
"""Comprehensive verification: function calls, adapters, vendors.

Runs each component with a minimal canned query and reports pass/fail.
Usage: python scripts/verify_all.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ======================================================================
# 1. Function Call Tools (via FunctionCallingRunner._call)
# ======================================================================

FUNCTION_TESTS = [
    ("pubmed_search",      {"query": "BRCA1 breast cancer", "max_results": 2}),
    ("gene_lookup",        {"gene_symbol": "BRCA1"}),
    ("clinvar_lookup",     {"variant": "BRCA1 c.5266dupC"}),
    ("rxnav_drug",         {"drug_name": "ibuprofen"}),
    ("openfda_adverse",    {"drug_name": "ibuprofen"}),
    ("dailymed_label",     {"drug_name": "ibuprofen"}),
    ("omim_lookup",        {"query": "Huntington disease"}),
    ("orphanet_lookup",    {"disease": "Huntington disease"}),
    ("medlineplus_topic",  {"topic": "diabetes"}),
    ("compute_calculator", {"calculator_name": "cha2ds2_vasc",
                             "params": {"age": 72, "sex": "female", "hypertension": True,
                                        "stroke_tia_history": True}}),
    ("calculator_eval",    {"formula": "(140-87)*48*1/(1.4*72)"}),
]


async def verify_function_calls():
    from harness.eval.function_calling_runner import FunctionCallingRunner
    from harness.llm_client import LLMClient

    # Dummy LLM (only used for .run() not for individual tools)
    llm = LLMClient(provider="openai", model="gpt-4o", api_key="test-api-key")
    runner = FunctionCallingRunner(llm=llm, per_tool_timeout=30)

    print("=" * 70)
    print("1. FUNCTION CALL TOOLS")
    print("=" * 70)
    results = []
    for name, args in FUNCTION_TESTS:
        start = time.monotonic()
        try:
            result = await runner._dispatch_tool(name, args)
            dt = time.monotonic() - start
            # A tool works if it returns non-empty content that isn't a pure error marker
            # Error markers: [tool_name timed out] or [tool_name error: ...]
            is_error = (
                not result
                or "timed out" in result.lower()
                or (result.startswith("[") and "error" in result.lower() and "]" in result[:60])
            )
            # OMIM without key returns a graceful skip message, count as "configurable" not fail
            is_configurable = "not configured" in (result or "").lower()
            if is_error:
                status = "✗"
            elif is_configurable:
                status = "○"
            else:
                status = "✓"
            has_content = not is_error
            preview = result[:80].replace("\n", " ") if result else "(empty)"
            print(f"  {status} {name:22s} [{dt:5.1f}s] {preview}")
            results.append((name, has_content, dt, preview))
        except Exception as exc:
            dt = time.monotonic() - start
            print(f"  ✗ {name:22s} [{dt:5.1f}s] EXCEPTION: {exc}")
            results.append((name, False, dt, str(exc)))
    return results


# ======================================================================
# 2. Adapters (instantiate + run with minimal query)
# ======================================================================

ADAPTER_TESTS = [
    # (adapter_name, query, context)
    ("ncbi_tools",            "BRCA1 gene function",                    {"genes": ["BRCA1"]}),
    ("geneagent",             "analyze gene set",                        {"genes": ["TP53", "BRCA1"]}),
    ("genegpt",               "what is BRCA1",                           None),
    ("genotex",               "gene expression in breast cancer",        {"trait": "breast cancer"}),
    ("genomas",               "analyze expression data",                 None),
    ("openbio",               "gene function",                           None),
    ("ehragent",              "patient labs",                            None),
    ("colacare",              "predict mortality",                       None),
    ("medagentbench",         "patient lookup",                          None),
    ("mdagents",              "diagnose chest pain",                     None),
    ("txagent",               "treat hypertension",                      None),
    ("agentclinic",           "diagnose case",                           None),
    ("torchxrv",              "analyze xray",                            None),
    ("medagent_pro",          "diagnose image",                          None),
    ("drugagent",             "drug discovery",                          None),
    ("prompt2pill",           "design drug",                             None),
    ("wearable",              "analyze sleep",                           {"wearable_data": {"avg_steps_30d": 5000}}),
    ("clinical_calculators",  "calculate BMI",                           {"calculator": "bmi",
                                                                          "calculator_params": {"weight_kg": 70, "height_m": 1.7}}),
    ("phenoage",              "biological age",                          {"age": 50, "biomarkers": {
        "albumin": 4.0, "creatinine": 1.0, "glucose": 100, "crp": 0.1,
        "lymphocyte_pct": 30, "mcv": 90, "rdw": 13, "alkaline_phosphatase": 60, "wbc": 6.0}}),
]


async def verify_adapters():
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
    from harness.adapters import ADAPTER_REGISTRY
    from harness.llm_client import LLMClient

    llm = LLMClient(provider="gemini", model="gemini-2.5-flash",
                     api_key=os.environ.get("GEMINI_API_KEY", "dummy"))

    # Build name -> class map
    name_to_class: dict[str, type] = {}
    # Map from registry by lowercasing class name
    from harness.adapters import (
        NCBIToolsAdapter, GeneAgentAdapter, GeneGPTAdapter, GenoTEXAdapter,
        GenoMASAdapter, OpenBioAdapter, EHRAgentAdapter, ColaCareAdapter,
        MedAgentBenchAdapter, MDAgentsAdapter, TxAgentAdapter, AgentClinicAdapter,
        TorchXRVAdapter, MedAgentProAdapter, DrugAgentAdapter, Prompt2PillAdapter,
        WearableAdapter, CalculatorAdapter, PhenoAgeAdapter,
    )
    name_to_class = {
        "ncbi_tools": NCBIToolsAdapter,
        "geneagent": GeneAgentAdapter,
        "genegpt": GeneGPTAdapter,
        "genotex": GenoTEXAdapter,
        "genomas": GenoMASAdapter,
        "openbio": OpenBioAdapter,
        "ehragent": EHRAgentAdapter,
        "colacare": ColaCareAdapter,
        "medagentbench": MedAgentBenchAdapter,
        "mdagents": MDAgentsAdapter,
        "txagent": TxAgentAdapter,
        "agentclinic": AgentClinicAdapter,
        "torchxrv": TorchXRVAdapter,
        "medagent_pro": MedAgentProAdapter,
        "drugagent": DrugAgentAdapter,
        "prompt2pill": Prompt2PillAdapter,
        "wearable": WearableAdapter,
        "clinical_calculators": CalculatorAdapter,
        "phenoage": PhenoAgeAdapter,
    }

    print()
    print("=" * 70)
    print("2. ADAPTERS")
    print("=" * 70)
    results = []
    for name, query, context in ADAPTER_TESTS:
        Cls = name_to_class.get(name)
        if Cls is None:
            print(f"  ✗ {name:22s} — not in registry")
            continue

        start = time.monotonic()
        # Use the adapter's default vendor path (don't override — class knows best)
        try:
            adapter = Cls(config={}, llm=llm)
            # Check availability
            if not adapter.available:
                dt = time.monotonic() - start
                print(f"  ○ {name:22s} [{dt:5.1f}s] unavailable: {adapter.unavailable_reason[:60]}")
                results.append((name, "unavailable", dt))
                continue

            # Run with timeout (60s to accommodate multi-LLM-call adapters)
            result = await asyncio.wait_for(adapter.run(query, context), timeout=60)
            dt = time.monotonic() - start
            answer = result.get("answer", "")
            confidence = result.get("confidence", 0)

            # Heuristic: confidence > 0.3 and non-trivial answer = passing
            is_real = confidence >= 0.3 and len(answer) > 30 and "not importable" not in answer and "fallback" not in answer.lower()
            status = "✓" if is_real else "○"
            preview = answer[:70].replace("\n", " ")
            print(f"  {status} {name:22s} [{dt:5.1f}s] conf={confidence:.2f} | {preview}")
            results.append((name, "real" if is_real else "fallback", dt))
        except asyncio.TimeoutError:
            dt = time.monotonic() - start
            print(f"  ✗ {name:22s} [{dt:5.1f}s] TIMEOUT")
            results.append((name, "timeout", dt))
        except Exception as exc:
            dt = time.monotonic() - start
            print(f"  ✗ {name:22s} [{dt:5.1f}s] EXCEPTION: {exc}")
            results.append((name, "error", dt))
    return results


# ======================================================================
# 3. Vendors (can we import / find entry point?)
# ======================================================================

VENDOR_TESTS = [
    # (vendor_name, entry_file, check_type)
    # check_type: "file" = entry file exists; "import" = try python -c import;
    ("GeneAgent",     "main_cascade.py",      "file"),
    ("GeneGPT",       "main.py",              "file"),  # may be different
    ("GenoTEX",       "README.md",            "file"),
    ("GenoMAS",       "README.md",            "file"),
    ("OpenBioLLM",    "README.md",            "file"),
    ("MedAgentBench", "README.md",            "file"),
    ("EhrAgent",      "README.md",            "file"),
    ("ColaCare",      "README.md",            "file"),
    ("MDAgents",      "main.py",              "file"),
    ("TxAgent",       "README.md",            "file"),
    ("AgentClinic",   "README.md",            "file"),
    ("MedAgent-Pro",  "README.md",            "file"),
    ("drugagent",     "README.md",            "file"),
    ("Prompt-to-Pill","README.md",            "file"),
]


def verify_vendors():
    print()
    print("=" * 70)
    print("3. VENDORS")
    print("=" * 70)

    vendors_dir = Path("vendors")
    results = []

    for name, entry_file, check_type in VENDOR_TESTS:
        vendor_path = vendors_dir / name
        if not vendor_path.exists():
            print(f"  ✗ {name:20s} — NOT CLONED")
            results.append((name, "not_cloned", False, False, False))
            continue

        entry = vendor_path / entry_file
        has_entry = entry.exists()

        # Check for requirements.txt
        has_reqs = (vendor_path / "requirements.txt").exists() or \
                    (vendor_path / "pyproject.toml").exists() or \
                    (vendor_path / "setup.py").exists()

        # Check for .venv_ready sentinel
        venv_ready = (vendor_path / ".venv_ready").exists()

        # Count Python files (rough viability indicator)
        py_files = list(vendor_path.rglob("*.py"))

        status = "✓" if has_entry and len(py_files) > 0 else "○"
        print(f"  {status} {name:20s} cloned={'Y':3s} entry={'Y' if has_entry else 'N'} "
              f"reqs={'Y' if has_reqs else 'N'} venv={'Y' if venv_ready else 'N'} "
              f"pyfiles={len(py_files):3d}")
        results.append((name, "cloned", has_entry, has_reqs, venv_ready))

    return results


# ======================================================================
# Main
# ======================================================================

async def main():
    fc_results = await verify_function_calls()
    ad_results = await verify_adapters()
    vd_results = verify_vendors()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    fc_pass = sum(1 for _, ok, *_ in fc_results if ok)
    print(f"Function calls:  {fc_pass}/{len(fc_results)} working")

    ad_pass = sum(1 for _, status, *_ in ad_results if status == "real")
    ad_fallback = sum(1 for _, status, *_ in ad_results if status == "fallback")
    ad_fail = len(ad_results) - ad_pass - ad_fallback
    print(f"Adapters:        {ad_pass} real | {ad_fallback} fallback | {ad_fail} failed  (of {len(ad_results)})")

    vd_cloned = sum(1 for _, s, *_ in vd_results if s == "cloned")
    vd_ready = sum(1 for _, _, has_entry, has_reqs, venv in vd_results if venv)
    print(f"Vendors:         {vd_cloned} cloned | {vd_ready} venv-ready  (of {len(vd_results)})")


if __name__ == "__main__":
    asyncio.run(main())
