"""Unit tests for the benchmark-aware harness config framework."""

from harness.benchmark_config import (
    BENCHMARK_CONFIGS,
    BenchmarkHarnessConfig,
    build_system_prompt,
    build_user_prompt,
    filter_tools,
    get_config,
    register_config,
)
from harness.tools.tool_categories import (
    TOOL_CATEGORIES,
    get_tools_by_category,
    get_tools_by_whitelist,
)


class TestBenchmarkConfig:
    def test_default_config_for_unknown(self):
        cfg = get_config("some_random_unregistered_benchmark")
        assert cfg.system_prompt_hint == ""
        assert cfg.tool_whitelist is None
        assert cfg.tool_categories == []
        assert cfg.expected_answer_format == ""
        assert cfg.enable_retrieval is True

    def test_register_and_retrieve(self):
        cfg = BenchmarkHarnessConfig(
            name="unit_test_bench",
            system_prompt_hint="Unit-test hint.",
            tool_whitelist=["alpha", "beta"],
            expected_answer_format="Return an integer.",
        )
        register_config(cfg)
        got = get_config("unit_test_bench")
        assert got.system_prompt_hint == "Unit-test hint."
        assert got.tool_whitelist == ["alpha", "beta"]
        assert got.expected_answer_format == "Return an integer."

    def test_registry_populated_from_package_import(self):
        # The package registers a bunch of real benchmarks on import.
        import harness.benchmark_configs  # noqa: F401
        assert "medcalc" in BENCHMARK_CONFIGS
        assert "labbench2" in BENCHMARK_CONFIGS
        assert "pathvqa" in BENCHMARK_CONFIGS

    def test_labbench2_has_expected_hint(self):
        import harness.benchmark_configs  # noqa: F401
        cfg = get_config("labbench2")
        assert "key passage" in cfg.system_prompt_hint.lower()

    def test_pathvqa_has_empty_whitelist(self):
        """pathvqa is vision-only; no tools should be advertised."""
        import harness.benchmark_configs  # noqa: F401
        cfg = get_config("pathvqa")
        assert cfg.tool_whitelist == []


class TestToolFiltering:
    def setup_method(self):
        self.all_tools = [
            {"function": {"name": "compute_calculator"}},
            {"function": {"name": "olsp_bgee_sparql"}},
            {"function": {"name": "pubmed_search"}},
            {"function": {"name": "olsp_uniprot_lookup"}},
            {"function": {"name": "some_unmapped_tool"}},
        ]

    def test_category_filter_calculation(self):
        out = get_tools_by_category(["calculation"], self.all_tools)
        names = [t["function"]["name"] for t in out]
        assert "compute_calculator" in names
        assert "pubmed_search" not in names

    def test_category_filter_multiple(self):
        out = get_tools_by_category(["protein", "literature"], self.all_tools)
        names = [t["function"]["name"] for t in out]
        assert "olsp_uniprot_lookup" in names
        assert "pubmed_search" in names
        assert "compute_calculator" not in names

    def test_category_filter_empty_returns_empty(self):
        assert get_tools_by_category([], self.all_tools) == []

    def test_whitelist_filter(self):
        out = get_tools_by_whitelist(["compute_calculator", "pubmed_search"],
                                       self.all_tools)
        names = [t["function"]["name"] for t in out]
        assert set(names) == {"compute_calculator", "pubmed_search"}

    def test_whitelist_filter_empty(self):
        assert get_tools_by_whitelist([], self.all_tools) == []


class TestBuildHelpers:
    def test_build_system_prompt_appends_hint_and_format(self):
        import harness.benchmark_configs  # noqa: F401
        out = build_system_prompt("medcalc", "BASE PROMPT")
        assert "BASE PROMPT" in out
        assert "Task-specific guidance" in out
        assert "compute_calculator" in out
        assert "Expected answer format" in out

    def test_build_system_prompt_unregistered_returns_base(self):
        out = build_system_prompt("nonexistent_xyz", "BASE PROMPT")
        assert out == "BASE PROMPT"

    def test_build_user_prompt_uses_format_task_prompt(self):
        # With labbench2 context present, the user prompt must include
        # the injected key passage (proving build_user_prompt delegates
        # to the context-injection helper when no custom formatter set).
        import harness.benchmark_configs  # noqa: F401
        task = {
            "question": "What reagent?",
            "context": {
                "benchmark": "labbench2",
                "sources": ["Paper Q"],
                "key_passage": "Reagent A was used at 1 mM.",
            },
        }
        out = build_user_prompt("labbench2", task)
        assert "Reagent A was used at 1 mM." in out
        assert "Paper Q" in out


class TestFilterTools:
    def setup_method(self):
        self.all_tools = [
            {"function": {"name": "compute_calculator"}},
            {"function": {"name": "olsp_bgee_sparql"}},
            {"function": {"name": "pubmed_search"}},
            {"function": {"name": "python_exec"}},
        ]

    def test_medcalc_filters_to_calculation_tools(self):
        import harness.benchmark_configs  # noqa: F401
        out = filter_tools("medcalc", self.all_tools)
        names = [t["function"]["name"] for t in out]
        assert "compute_calculator" in names
        assert "olsp_bgee_sparql" not in names

    def test_pathvqa_empty_whitelist(self):
        import harness.benchmark_configs  # noqa: F401
        out = filter_tools("pathvqa", self.all_tools)
        assert out == []

    def test_unregistered_returns_all(self):
        out = filter_tools("nonexistent_benchmark", self.all_tools)
        assert len(out) == 4


class TestToolCategoriesCoverage:
    def test_known_tools_have_categories(self):
        """Sanity check: the core tools we rely on are tagged."""
        for tool in ("compute_calculator", "pubmed_search",
                      "olsp_uniprot_lookup", "python_exec"):
            assert tool in TOOL_CATEGORIES, f"{tool} missing category tags"
