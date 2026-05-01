from __future__ import annotations

from harness.cli import BACKBONES, BENCHMARKS, MODES
from harness.eval.hf_benchmark_registry import (
    HF_VERIFIED_BENCHMARK_KEYS,
    hf_verified_metadata,
    validate_hf_metadata,
)
from harness.tools import TOOL_SPECS
from harness.tools.tool_categories import (
    OPTIONAL_TOOL_CATEGORY_ENTRIES,
    TOOL_CATEGORIES,
    uncategorised_tools,
)


def _tool_name(spec: dict) -> str:
    if isinstance(spec.get("function"), dict):
        return spec["function"].get("name", "")
    return spec.get("name", "")


def test_public_registry_counts_meet_release_surface():
    assert len(BENCHMARKS) >= 100
    assert len(TOOL_SPECS) == 75
    assert len(MODES) == 4
    assert len(BACKBONES) >= 3


def test_hf_verified_metadata_contract_is_complete():
    assert validate_hf_metadata() == []
    metadata = hf_verified_metadata()
    assert set(metadata) == set(HF_VERIFIED_BENCHMARK_KEYS)
    assert len(metadata) >= 100
    for key, meta in metadata.items():
        assert meta["key"] == key
        assert meta["source_url"].startswith("https://huggingface.co/datasets/")
        assert meta["status"] == "verified"
        assert meta["input_type"] in {"text", "image+text", "structured"}
        assert meta["answer_type"] in {
            "multipleChoice",
            "exactMatch",
            "exactNumeric",
            "openText",
        }


def test_tool_registry_contract_is_stable():
    names = [_tool_name(spec) for spec in TOOL_SPECS]
    assert len(names) == len(set(names))
    assert uncategorised_tools(TOOL_SPECS) == []

    for spec in TOOL_SPECS:
        name = _tool_name(spec)
        fn = spec.get("function", spec)
        assert name
        assert fn.get("description"), name
        assert "parameters" in fn, name

    stale = set(TOOL_CATEGORIES) - set(names) - set(OPTIONAL_TOOL_CATEGORY_ENTRIES)
    assert stale == set()
