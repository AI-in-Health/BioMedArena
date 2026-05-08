# API reference

Python API for programmatic access to BioMedArena. For
command-line usage see the ``biomedarena`` CLI documented in the
top-level ``README.md``.

## Entry points

Primary entry points:

- ``harness.eval.benchmark_suite.BenchmarkSuite`` — benchmark
  evaluation across modes. Use this for running benchmarks
  programmatically.
- ``harness.cli`` / ``biomedarena`` — public command-line interface
  for single cells, benchmark listing, model listing, and mode listing.
- ``harness.eval.bench_*`` loaders — benchmark-specific loaders that
  normalize upstream datasets into the common task shape.

---

## `BenchmarkSuite`

```python
from harness.eval.benchmark_suite import BenchmarkSuite

suite = BenchmarkSuite(config_path="config.yaml")
```

### Constructor

```python
BenchmarkSuite(config_path: str = "config.yaml")
```

- ``config_path``: path to a YAML file with at least an ``llm``
  block. ``${ENV_VAR}`` substitutions are expanded at load time.
  Example:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5
  api_key: ${ANTHROPIC_API_KEY}
```

### `eval_tasks`

```python
async def eval_tasks(
    benchmark_name: str,
    tasks: list[dict],
    mode: str,
    max_concurrent: int = 5,
    benchmark_key: str | None = None,
    backbone_id: str | None = None,
    trace_sink: callable | None = None,
    enable_thinking: bool | None = None,
) -> BenchmarkMetrics
```

Evaluate a list of tasks in a given mode. Each task dict must have
the keys:

- ``id``: task identifier (str)
- ``question``: task prompt (str)
- ``answer``: gold answer (type depends on ``answer_type``)
- ``answer_type``: one of ``multipleChoice``, ``exactNumeric``,
  ``exactMatch``, ``openText``, ``keyword``, ``regex``
- ``category``: coarse-grained classification (str)

Optional keys:

- ``context``: dict of task-specific metadata used by context injection
- ``raw_subject``: free-form domain tag
- ``choices``: for MCQ tasks, a list of answer choices
- ``_benchmark_key``: set automatically to ``benchmark_key`` at
  dispatch time; the runner uses it to pick the right iteration budget

Additional parameters:

- ``backbone_id``: model identifier string for trace metadata
- ``trace_sink``: callback ``(task_dict, result) -> None`` invoked
  after each task completes; used by the matrix runner to dump
  per-task trace JSON
- ``enable_thinking``: override provider-level reasoning (``True``
  enables extended thinking, ``False`` disables, ``None`` uses
  mode default)

``mode`` is one of: ``simple_llm``, ``deep_think``,
``light``, ``heavy``, or ``self_consistency:<inner_mode>``.

``benchmark_key`` (default equals ``benchmark_name``) selects the
per-benchmark iteration budget in ``DEFAULT_MAX_ITERATIONS``. Use a
more specific key for sub-tasks that need distinct budgets (e.g.
``labbench_litqa2`` vs ``labbench_cloning_scenarios``).

Returns a ``BenchmarkMetrics`` object with:

- ``per_question``: list of ``QuestionMetric``
- ``task_success_rate``: fraction correct
- ``tool_call_accuracy``: average fraction of useful tool calls
- ``reasoning_faithfulness``: average reasoning faithfulness score
- ``avg_latency_s``
- ``success_by_category``: per-category success rates
- ``success_by_type``: per-type success rates

### `QuestionMetric` fields

```python
@dataclass
class QuestionMetric:
    question_id: str
    benchmark: str
    mode: str
    category: str
    question_text: str
    expected: str
    predicted: str
    predicted_raw: str
    task_success: bool
    tool_calls_made: list[str]
    tool_call_accuracy: float = 0.0
    reasoning_faithfulness: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
```

---

## Benchmark loaders

All loaders live under ``harness.eval`` and follow the signature
``load_<name>_tasks(limit=N, **kwargs) -> list[dict]``. Each returned
task dict conforms to the shape documented in ``eval_tasks`` above.

| Loader | Module |
|---|---|
| `load_aa_lcr_tasks` | `harness.eval.bench_aa_lcr` |
| `load_agentclinic_tasks` | `harness.eval.bench_agentclinic` |
| `load_bioasq_tasks` | `harness.eval.bench_bioasq` |
| `load_bioprobench_tasks` | `harness.eval.bench_bioprobench` |
| `load_bixbench_tasks` | `harness.eval.bench_bixbench` |
| `load_genotex_tasks` | `harness.eval.bench_genotex` |
| `load_gpqa_bio_tasks` | `harness.eval.bench_gpqa_bio` |
| `load_healthbench_tasks` | `harness.eval.bench_healthbench` |
| `load_hf_benchmark_tasks` | `harness.eval.bench_hf_benchmark` (`hf_*` registry entries) |
| `load_hle_gold_tasks` | `harness.eval.bench_hle_gold` |
| `load_labbench_tasks` | `harness.eval.bench_labbench` |
| `load_labbench2_tasks` | `harness.eval.bench_labbench2` |
| `load_medagentbench_tasks` | `harness.eval.bench_medagentbench` |
| `load_medcalc_tasks` | `harness.eval.bench_medcalc` |
| `load_medhelm_tasks` | `harness.eval.bench_medhelm` |
| `load_medical_qa_tasks` | `harness.eval.bench_medical_qa` (`medqa`, `medmcqa`, `pubmedqa`, and compatibility aliases) |
| `load_medxpertqa_tasks` | `harness.eval.bench_medxpertqa` |
| `load_medxpertqa_mm_tasks` | `harness.eval.bench_medxpertqa_mm` |
| `load_mmlu_tasks` | `harness.eval.bench_mmlu` |
| `load_pathvqa_tasks` | `harness.eval.bench_pathvqa` |
| `load_quick_suite_tasks` | `harness.eval.bench_quick_suite` |
| `load_rag_essential_tasks` | `harness.eval.bench_rag_essential` |
| `load_super_chemistry_tasks` | `harness.eval.bench_super_chemistry` |
| `load_superchem_tasks` | `harness.eval.bench_superchem` |
| `load_supergpqa_tasks` | `harness.eval.bench_supergpqa` |

Common kwargs: ``limit`` (max tasks), ``seed`` (for deterministic
sampling), benchmark-specific filters (e.g. ``subsets`` for LAB-Bench
and LAB-Bench 2, ``subset`` for MedXpertQA, ``include_chemistry`` for
HLE-Gold).

---

## `BenchmarkHarnessConfig` and registry

```python
from harness.benchmark_config import (
    BenchmarkHarnessConfig, register_config, get_config, filter_tools,
)
```

### `BenchmarkHarnessConfig` fields

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Unique benchmark key |
| `system_prompt_hint` | `str` | Appended to the system prompt |
| `tool_categories` | `list[str]` | Category tags for grouped tool whitelisting |
| `tool_whitelist` | `list[str] \| None` | Exact tool-name whitelist (overrides categories if set) |
| `context_formatter` | `Callable[[dict], str] \| None` | Custom function to format task context into prompt text |
| `expected_answer_format` | `str` | One-line description of expected answer format |
| `enable_retrieval` | `bool` | Whether the retrieval subsystem is enabled (default `True`) |

### Registering a custom config

```python
register_config(BenchmarkHarnessConfig(
    name="my_custom_benchmark",
    system_prompt_hint="You are solving custom tasks.",
    tool_categories=["literature", "search"],
    expected_answer_format="A single letter.",
))
```

### Retrieving a config

```python
cfg = get_config("medcalc")  # returns None if unregistered
```

### Filtering the tool pool

```python
from harness.eval.function_calling_runner import TOOL_SPECS
filtered = filter_tools("medcalc", TOOL_SPECS)
# -> list of tools whose names match the registered category or whitelist
```

---

## Command-line entry point

The installed console script is ``biomedarena``:

```bash
biomedarena list-benchmarks
biomedarena list-backbones
biomedarena list-modes
biomedarena run --benchmark medcalc --backbone gemini-2.5-flash --limit 5
```

The CLI dispatches through ``harness.cli.main`` and writes optional
per-task JSON summaries with ``--output``.

---

## Cost monitor

```python
from harness.cost_monitor import snapshot

usage = snapshot()
# {"total_usd": float, "per_provider": {...}, "per_model": {...}}
```

The cost monitor is stamped automatically by ``LLMClient``. A
persistent ledger is maintained at ``data/cost_ledger.json``.
