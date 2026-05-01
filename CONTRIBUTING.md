# Contributing

Contributions are welcome! Please follow the guidelines below.

## Getting set up

1. Clone the repository.
2. Create a Python 3.11+ virtualenv: ``python -m venv .venv``.
3. ``pip install -e ".[dev,eval,provider-gemini]"``.
4. Copy ``.env.example`` to ``.env`` and fill in keys.
5. Run the unit suite: ``pytest tests/unit -v``.

## Branching and commits

- Branch from ``main``.
- Keep commits focused — one logical change per commit.
- Commit messages: short summary (<70 chars) on the first line,
  optional body after a blank line.
- Rebase onto ``main`` before opening a PR.

## Testing

Unit tests must pass before merge:

```bash
pytest tests/unit -v
```

Smoke tests are opt-in (require API keys or running services):

```bash
pytest tests/smoke -v
```

When adding a new benchmark or tool, include at least one unit test
for the loader or schema and (if feasible) one smoke test.

## Adding a benchmark

1. Create ``harness/eval/bench_<name>.py`` with a
   ``load_<name>_tasks(limit=N, **kwargs)`` function.
2. Each returned task must have ``id``, ``question``, ``answer``,
   ``answer_type``, ``category``.
3. Register the loader import in ``harness/eval/__init__.py``.
4. Register a ``BenchmarkHarnessConfig`` in
   ``harness/benchmark_configs/registry.py``.
5. Add the benchmark to ``harness/cli.py::BENCHMARKS`` (for CLI
   access) and to ``configs/matrix_default.yaml`` (for matrix runs).
6. Document source, task type, sample size, scorer, and any
   benchmark-specific caveats in ``docs/benchmark_datasets.md``.

## Adding a tool

1. If the tool is a thin HTTP/SPARQL wrapper for a public endpoint,
   prefer the ``olsp_*`` pattern in
   ``harness/tools/openai_ported/`` — see ``NOTICE.md`` for the
   conventions.
2. If it wraps a Python library, put the handler in
   ``harness/tools/<domain>_tools.py`` and register the TOOL_SPEC in
   ``harness/eval/function_calling_runner.py``.
3. Tag the tool in ``harness/tools/tool_categories.py`` so
   benchmark-aware filtering picks it up.
4. Add a unit test for the schema and a smoke test that exercises a
   trivial happy path.

## Documentation

- Update ``README.md`` only for first-screen setup or usage changes.
- Update ``docs/benchmark_datasets.md`` when you add or change a
  benchmark.
- Update ``docs/tools_and_skills.md`` when you add or change a tool.
- Keep ``docs/api_reference.md`` in sync with public API changes.

## Code style

- Type hints on public functions.
- Async is the default for LLM and tool entry points; keep sync code
  for pure computation.
- Logging via ``logging.getLogger(__name__)``. Do not add status
  emoji in code or commit messages.

## Questions

Open a GitHub issue or discussion in the project repository.
