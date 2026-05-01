# BioMedArena

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">HLE-Verified-Gold (Biomed+Chem)</th>
      <th colspan="2">BixBench</th>
      <th colspan="2">LAB-Bench 2</th>
      <th colspan="2">Super Chemistry</th>
    </tr>
    <tr>
      <th>Baseline</th>
      <th>Ours</th>
      <th>Baseline</th>
      <th>Ours</th>
      <th>Baseline</th>
      <th>Ours</th>
      <th>Baseline</th>
      <th>Ours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SOTA</td>
      <td colspan="2">46.8</td>
      <td colspan="2">80.5</td>
      <td colspan="2">80.0</td>
      <td colspan="2">38.5</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/arcee-ai/Trinity-Large-Thinking">Trinity-Large-Thinking</a></td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16">NVIDIA Nemotron-3 Super 120B</a></td>
      <td>10.7</td><td>29.5</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/PrimeIntellect/INTELLECT-3.1">INTELLECT-3.1</a></td>
      <td>17.4</td><td>24.2</td>
      <td>5.4</td><td>20.0</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/zai-org/GLM-4.5">GLM-4.5</a></td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3.5-397B-A17B">Qwen3.5-397B-A17B</a></td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.anthropic.com/news/claude-sonnet-4-5">Claude Sonnet 4.5</a></td>
      <td>20.8</td><td>41.6</td>
      <td>17.1</td><td>42.4</td>
      <td>48.1</td><td>71.3</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.anthropic.com/news/claude-opus-4-5">Claude Opus 4.5</a></td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.anthropic.com/research/claude-sonnet-4-6">Claude Sonnet 4.6</a></td>
      <td>23.5</td><td>43.6</td>
      <td>40.5</td><td>48.3</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://www.anthropic.com/news/claude-opus-4-6">Claude Opus 4.6</a></td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://developers.openai.com/api/docs/models/gpt-5.4">GPT-5.4</a></td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://blog.google/products/gemini/gemini-3-flash/">Gemini 3 Flash</a></td>
      <td>38.26</td><td>50.34</td>
      <td>38.54</td><td>69.27</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
    <tr>
      <td><a href="https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/">Gemini 3.1 Pro</a></td>
      <td>-</td><td>-</td>
      <td>43.41</td><td>85.85</td>
      <td>-</td><td>-</td>
      <td>-</td><td>-</td>
    </tr>
  </tbody>
</table>

BioMedArena is a biomedical agent evaluation harness for comparing
LLM backbones, tool-use modes, scorers, and datasets behind one CLI.
It currently has 147 registered benchmarks, 75 tools, 4 modes, and
8 registered model IDs.

The project is designed as a practical research surface: add a dataset,
choose a harness mode, expose a tool pack, run a matrix, and compare
whether agentic behavior actually improves biomedical, medical,
chemistry, biology, protein, genomics, DNA/RNA, and healthcare tasks.

## Quick Check

After installing dependencies, run the offline smoke suite:

```bash
python3 scripts/run_quick_suite.py
```

Expected healthy output:

- 147 registered benchmarks
- 75 registered tools
- 4 registered modes
- 20/20 scorer checks passed

For the stricter offline release gate:

```bash
python3 scripts/release_gate.py --strict
```

## Installation

```bash
git clone https://github.com/AI-in-Health/BioMedArena.git
cd BioMedArena

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev,eval,provider-gemini]"

cp .env.example .env
```

Fill at least one model provider key in `.env`:

```bash
OPENAI_API_KEY=<your-openai-api-key>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
GEMINI_API_KEY=<your-gemini-api-key>
HF_TOKEN=<your-huggingface-token-for-gated-benchmarks>
```

Gated HuggingFace datasets also require accepting the dataset terms in
the browser before `HF_TOKEN` can load them. See `.env.example` for
optional domain-specific keys such as NCBI, OMIM, Serper, and Jina.

## Basic Usage

List available resources:

```bash
bioagent list-benchmarks
bioagent list-backbones
bioagent list-modes
```

The package name is `biomedarena`; the command-line entry point remains
`bioagent` for compatibility. Environment variables use the `BIOAGENT_`
prefix for the same reason.

Run one benchmark cell:

```bash
bioagent run \
  --benchmark medcalc \
  --backbone gemini-2.5-flash \
  --tools biomed --reasoning-mode light \
  --limit 5 \
  --output result.json
```

Run a small matrix cell:

```bash
python3 scripts/run_matrix.py \
  --config configs/matrix_default.yaml \
  --only medcalc,gemini,simple_llm \
  --limit-override 1
```

Check official source accessibility before spending model budget:

```bash
python3 scripts/verify_benchmark_sources.py --benchmarks all
```

## Execution Modes

The public CLI exposes four modes:

| Mode | Purpose |
| --- | --- |
| `simple_llm` | Pure model baseline, no tools. |
| `deep_think` | Native model reasoning/thinking path where supported. |
| `light` | Single-turn function/tool calling. |
| `heavy` | Multi-turn ReAct loop with tool retrieval. |

A unified CLI interface is also available via `--tools` / `--reasoning-mode` /
`--enable-thinking` flags, which map to the modes above:

| `--tools` | `--reasoning-mode` | Internal mode | Thinking |
| --- | --- | --- | --- |
| `off` | (n/a) | `deep_think` | ON (default) |
| `off` + `--enable-thinking 0` | (n/a) | `simple_llm` | OFF |
| `biomed` / `search` / `all` | `light` | `light` | OFF |
| `biomed` / `search` / `all` | `heavy` | `heavy` | ON |

The legacy `--mode` / `--web-tools` flags remain supported for backward
compatibility. Add `--self-consistency` to wrap any mode with majority voting.

## Documentation

The root README stays short on purpose. Detailed release information
lives in `docs/`:

- [Benchmark dataset inventory](docs/benchmark_datasets.md)
- [Tools and skills inventory](docs/tools_and_skills.md)
- [API reference](docs/api_reference.md)
- [Metrics guide](docs/metrics_guide.md)

## Security

`python_exec` can execute model-supplied Python with timeout and basic
denylist checks. Treat this as a convenience guard, not a hardened
sandbox. Run untrusted workloads in an isolated container or VM, keep
secrets out of the working directory, and disable code-execution or
web-search tools for private data unless you have reviewed the policy.

External tools may call third-party APIs and public databases. Review
the benchmark and tool inventories before running sensitive workloads.

## Testing

```bash
python3 scripts/run_quick_suite.py
python3 scripts/release_gate.py --strict
python3 -m pytest tests/unit -q
python3 -m pytest tests/smoke -q -m "not slow"
```

## License

See [LICENSE](LICENSE). Ported life-science skill attribution is tracked
in [harness/tools/openai_ported/NOTICE.md](harness/tools/openai_ported/NOTICE.md).
