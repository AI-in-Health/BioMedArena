# Benchmark pipeline metrics guide

How to run benchmarks (single cell or full matrix), how results are
collected, and how every column in the 22-field results CSV is computed.

## Table of contents

1. [Running benchmarks](#running-benchmarks)
2. [CSV schema](#csv-schema)
3. [Column definitions](#column-definitions)
4. [Per-task traces](#per-task-traces)
5. [LLM-as-judge scoring](#llm-as-judge-scoring)
6. [Supported models and modes](#supported-models-and-modes)

---

## Running benchmarks

### Single cell via the CLI

```bash
biomedarena run \
    --benchmark medcalc \
    --backbone claude-sonnet-4-6 \
    --mode deep_think \
    --limit 10
```

Output: per-task results printed to stdout plus (if `--output`
specified) a per-task JSON summary. Use `scripts/run_matrix.py`
(below) to get the 22-column CSV and per-task traces.

### Matrix runner

Define a YAML at `configs/<name>.yaml`:

```yaml
run_name: custom_run

backbones:
  - id: claude
    provider: anthropic
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
  - id: gemini
    provider: gemini
    model: gemini-2.5-flash
    api_key_env: GEMINI_API_KEY

modes_default: [simple_llm, deep_think]

benchmarks:
  - name: medcalc
    loader: load_medcalc_tasks
    kwargs: {limit: 30}
    benchmark_key: medcalc
  - name: labbench
    loader: load_labbench_tasks
    kwargs: {limit: 10, subsets: ["LitQA2", "ProtocolQA"]}
    benchmark_key: labbench

seeds: [42]

cost_monitor:
  hard_stop_usd: 50
  soft_warn_usd: 30

timeout:
  per_task_s: 240
  per_cell_s: 1800

base_config: config.yaml
```

Save the example as `configs/custom_matrix.yaml`, or use the checked-in
starter matrix directly:

```bash
python3 scripts/run_matrix.py --config configs/matrix_default.yaml
```

### Runner flags

| Flag | Purpose |
|---|---|
| `--config PATH` | matrix YAML (default `configs/matrix_default.yaml`) |
| `--run-name NAME` | directory prefix (default: config `run_name` or `matrix`) |
| `--output-dir PATH` | explicit output directory (overrides `--run-name + timestamp`) |
| `--limit-override N` | force every benchmark's `kwargs.limit` to `N` (useful for dry runs) |
| `--only` | filter to specific cells: `benchmark,backbone,mode;benchmark,backbone,mode;...` |
| `--patch --benchmarks X,Y` | append specific benchmarks to the latest existing `data/runs/matrix_*` dir |

### Output layout

```
data/runs/<run_name>_<timestamp>/
├── <backbone_id>__<mode>__<benchmark>.json      # per-cell summary
├── MATRIX_SUMMARY.json                          # cell-level aggregates
├── results.csv                                  # 22-column per-cell rows
└── traces/
    └── <backbone_id>__<mode>__<benchmark>/
        └── <task_id>.json                       # full trace per task
```

No existing directory is ever overwritten: each run gets a fresh
timestamped subfolder.

### Avoiding duplicate runs

`--patch` only appends *new* cells whose per-cell JSON doesn't yet
exist. To genuinely re-run a cell, delete its per-cell JSON first.

---

## CSV schema

Each run produces `results.csv` with **22 columns** and one row per
cell (`benchmark × backbone × mode × seed`).

### Column order (preserve when reading)

```
benchmark, backbone, mode, seed,
n_tasks, n_correct, n_error, accuracy,
total_cost_usd, avg_cost_per_task_usd,
total_input_tokens, total_output_tokens,
avg_input_tokens, avg_output_tokens,
avg_tool_calls, avg_iterations, tool_call_success_rate,
unique_tools_used, top_tool, top_tool_call_count,
avg_latency_s, timestamp
```

---

## Column definitions

Each column is computed from the list of `TraceRecorder` objects for
the tasks in the cell.

| Column | Computed as | Notes |
|---|---|---|
| `benchmark` | `task.benchmark_name` | E.g., `medcalc` |
| `backbone` | Config backbone string | E.g., `claude-sonnet-4-6` |
| `mode` | Harness mode | simple_llm / deep_think / light / heavy |
| `seed` | Config `seeds[0]` | Default `42` |
| `n_tasks` | `len(traces)` | Sample size N |
| `n_correct` | `sum(t.scorer_result.correct for t in traces)` | k |
| `n_error` | `sum(1 for t in traces if any(c.error for c in t.llm_calls))` | Runtime errors, not scorer failures |
| `accuracy` | `n_correct / n_tasks` | Strict: errors count as wrong. 0.0 if n_tasks=0 |
| `total_cost_usd` | `sum(t.total_cost_usd() for t in traces)` | All LLM calls in the cell |
| `avg_cost_per_task_usd` | `total_cost_usd / n_tasks` | |
| `total_input_tokens` | `sum(t.total_input_tokens() for t in traces)` | Sum over all LLM calls |
| `total_output_tokens` | `sum(t.total_output_tokens() for t in traces)` | Includes thinking tokens for `deep_think` |
| `avg_input_tokens` | `total_input_tokens / n_tasks` | |
| `avg_output_tokens` | `total_output_tokens / n_tasks` | |
| `avg_tool_calls` | `total_tool_calls / n_tasks` | simple_llm and deep_think are 0 by contract |
| `avg_iterations` | `total_iterations / n_tasks` | simple_llm=1, deep_think=1, light=1–N, heavy=2 |
| `tool_call_success_rate` | `n_tool_ok / n_tool_calls` | 1.0 if no tool calls |
| `unique_tools_used` | `len({c.name for c in all_tool_calls})` | Distinct tool names |
| `top_tool` | Most-called tool name | `""` if no tool calls |
| `top_tool_call_count` | Count of top tool | 0 if no tool calls |
| `avg_latency_s` | `sum(t.total_latency_s) / n_tasks` | End-to-end wall time per task |
| `timestamp` | Cell completion time | ISO 8601 UTC |

### Iteration counting convention

- `simple_llm` — 1 (single LLM call)
- `deep_think` — 1 (single LLM call with provider reasoning budget)
- `light` — 1 per ReAct loop iteration (typically 1–N where N=max_iterations)
- `heavy` — Triage + Reason = 2 (or 1 if the confidence gate falls back to deep_think)
- `self_consistency:*` — inherits the chosen inner mode

### Edge case handling

- `n_tasks = 0` → `accuracy = 0.0` (never NaN)
- No tool calls → `avg_tool_calls=0`, `tool_call_success_rate=1.0`, `unique_tools_used=0`, `top_tool=""`, `top_tool_call_count=0`
- Provider does not return token usage → treated as 0
- If a cell fails entirely, its row is omitted; the per-cell JSON still lands on disk with an error note

---

## Per-task traces

Every task produces a detailed JSON trace at
`data/runs/<run>/traces/<cell_name>/<task_id>.json`.

Contents:

| Field | Meaning |
|---|---|
| `task_id`, `benchmark`, `backbone`, `mode` | Identifiers |
| `started_at`, `finished_at`, `total_latency_s` | Timing |
| `iterations` | ReAct-loop count (see convention above) |
| `llm_calls[]` | Each LLM turn: `system`, `messages` (truncated), `response_text`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`, `finish_reason`, `error` |
| `tool_calls[]` | Each tool invocation: `iteration`, `name`, `arguments` (truncated), `result_preview` (first 500 chars), `success`, `error`, `latency_ms` |
| `final_answer` | Truncated to 4000 chars |
| `scorer_result` | `correct`, `method`, `details` (including `primary_verdict`, `primary_method`, `is_open_ended`, `judge_invoked`, `judge_verdict`, `judge_model`) |

Traces are excluded from git (via `.gitignore`) because they contain
raw LLM output that may include copyrighted benchmark question text.

Long messages and tool arguments are truncated to keep trace files
reasonable in size (4KB per message, 2KB per content block, 1KB per
tool argument).

---

## LLM-as-judge scoring

The scoring pipeline has two routes depending on the task
`answer_type`. Implemented in `harness/eval/llm_judge.py::score_with_fallback`.

### Route A — MCQ / structured (judge as fallback)

Applies to: `answer_type` ∈ `{multipleChoice, numeric, exact, exactNumeric, exactMatch}`, plus any task with no explicit `answer_type`.

1. Primary scorer (letter match, numeric tolerance, regex) runs.
2. If primary says **correct** → final = correct. Judge is **not** called.
3. If primary says **incorrect** and the candidate is empty → final = incorrect. Judge is **not** called (no point).
4. If primary says **incorrect** and the candidate is non-empty → LLM judge is invoked.
    - Judge **correct** → promote to correct. `method = primary:<type>+llm_judge_fallback(<model>)`
    - Judge **incorrect** → stay incorrect. `method` unchanged.
5. **The judge can only promote. It can never demote.** Re-scoring an existing dataset with this path is monotonically non-decreasing.

### Route B — Open-ended (judge as primary)

Applies to: `answer_type` ∈ `{openText, openEnded, freeText}`.

1. Primary deterministic scorer still runs, but its verdict is **metadata only**. It is preserved in `scorer_result.details.primary_verdict`.
2. LLM judge is invoked as the **primary scorer**.
3. Judge verdict is authoritative: `method = llm_judge_primary(<model>)`.

### Judge model

The judge model is configurable per run. By default, BioMedArena uses
`claude-sonnet-4-5`, but users can pin another supported judge with
`BIOMEDARENA_JUDGE_PROVIDER` and `BIOMEDARENA_JUDGE_MODEL`. Using a
single configured judge across a run keeps cross-run comparisons stable
and removes target-dependent variability.

Cost implication: judge calls are more expensive than the previous
dynamic policy (~10x per judge call). The judge is only invoked
under Route A when the primary scorer says incorrect AND the
candidate is non-empty, and under Route B for every open-ended
task. Typical runs see judge cost as a small fraction of total LLM
spend.

### Enable / disable

```bash
# Default: judge is ENABLED
python3 scripts/run_matrix.py --config configs/matrix_default.yaml

# Optional: pin a custom judge model for the whole run
BIOMEDARENA_JUDGE_PROVIDER=gemini \
BIOMEDARENA_JUDGE_MODEL=gemini-3-flash-preview \
python3 scripts/run_matrix.py --config configs/matrix_default.yaml

# Disable globally
BIOMEDARENA_LLM_JUDGE=0 python3 scripts/run_matrix.py --config configs/matrix_default.yaml
```

Every trace records `scorer_result.details.judge_invoked` and
`judge_verdict` so disagreement analysis is trivial post-hoc.

### Failure handling

If the judge API call errors, the primary verdict is kept and the
error is recorded in `scorer_result.details.judge_error`. Judge
failure never aborts a benchmark run.

---

## Supported models and modes

### Backbones

| Model ID | Provider | Notes |
|---|---|---|
| claude-sonnet-4 | Anthropic | |
| claude-sonnet-4-5 | Anthropic | Judge default |
| claude-sonnet-4-6 | Anthropic | Current |
| claude-opus-4-5 | Anthropic | |
| claude-opus-4-6 | Anthropic | Strongest Anthropic |
| gpt-4o | OpenAI | General OpenAI chat model |
| gemini-2.5-flash | Google | Low-cost Google model |
| gemini-2.5-pro | Google | |
| gemini-3-flash-preview | Google | Current Gemini Flash preview |

List at runtime: `biomedarena list-backbones`. The built-in CLI
registry currently covers API-backed provider models; local open-source
or self-hosted models should be added through a provider adapter before
they appear in this list.

### Modes

| Mode | Description | Tools? | Iterations |
|---|---|---|---|
| simple_llm | Single LLM call | no | 1 |
| deep_think | Single LLM call with provider reasoning budget | no | 1 |
| light | Single-turn tool calling (1–N rounds) | yes | 1–N (per_benchmark `DEFAULT_MAX_ITERATIONS`) |
| heavy | Multi-turn ReAct with tool retrieval | yes | 2 (or 1 if confidence gate escalates to deep_think) |

`self_consistency:<inner>` wraps any of the above modes with majority
voting (N samples, configurable via config YAML). Enable via CLI with
`--self-consistency`.

List at runtime: `biomedarena list-modes`.
