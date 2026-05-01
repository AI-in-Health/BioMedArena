#!/usr/bin/env bash
# =============================================================================
# quick_run.sh --- BioMedArena Quick Start (7 experiments)
#
# A minimal experiment suite for new users to reproduce our key comparisons.
# Covers the core axes: thinking baseline, domain tools, web search, combined,
# and light (light) vs heavy (heavy) modes.
#
# Usage:
#   ./quick_run.sh <backbone> <benchmark> [limit]
#
# Examples:
#   ./quick_run.sh claude-sonnet-4-6 hle_gold
#   ./quick_run.sh claude-sonnet-4-6 hle_gold 20
#   ./quick_run.sh gpt-4o medcalc 30
#   ./quick_run.sh gemini-2.5-flash bixbench
#
# Prerequisites:
#   - pip install -e ".[eval]"
#   - .env with at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
#   - SERPER_API_KEY required for settings 4-7 (web search)
#   - HF_TOKEN required for gated benchmarks (hle_gold, labbench2, labbench)
#
# Experiment matrix (7 settings):
#
# +---------+-------------------+----------+----------------+-------------------+
# | Setting | Name              | --tools  | --reason-mode  | Context Mgmt      |
# +---------+-------------------+----------+----------------+-------------------+
# |    1    | Thinking baseline | off      | (n/a)          | None              |
# |    2    | Domain (light)    | biomed   | light          | Scratchpad        |
# |    3    | Domain (heavy)    | biomed   | heavy          | Scratchpad        |
# |    4    | Search (light)    | search   | light          | Scratchpad        |
# |    5    | Search (heavy)    | search   | heavy          | Scratchpad        |
# |    6    | Combined (light)  | all      | light          | Scratchpad        |
# |    7    | Combined (heavy)  | all      | heavy          | Scratchpad + Inc  |
# +---------+-------------------+----------+----------------+-------------------+
#
# light = single-turn function calling (faster / cheaper)
# heavy = multi-turn ReAct loop (deeper / stronger)
#
# Key comparisons enabled:
#   - Thinking vs tool use:    setting 1 vs 2
#   - Light vs heavy:          setting 2 vs 3, 4 vs 5, 6 vs 7
#   - Domain vs search:        setting 2 vs 4, 3 vs 5
#   - Domain vs combined:      setting 2 vs 6, 3 vs 7
#
# Output: results/<benchmark>/<backbone>/
# =============================================================================

set -euo pipefail

# ----------------------------- Args -----------------------------------------

BACKBONE="${1:?Usage: $0 <backbone> <benchmark> [limit]}"
BENCHMARK="${2:?Usage: $0 <backbone> <benchmark> [limit]}"
LIMIT="${3:-9999}"

# ----------------------------- Directories ----------------------------------

OUTDIR="results/${BENCHMARK}/${BACKBONE}"
mkdir -p "$OUTDIR"

# ----------------------------- Common flags ---------------------------------

COMMON=(
    bioagent run
    --benchmark "$BENCHMARK"
    --backbone "$BACKBONE"
    --limit "$LIMIT"
    --verbose
    --resume
)

# ----------------------------- Helper ---------------------------------------

run_exp() {
    local name="$1"
    shift
    local outfile="${OUTDIR}/${name}.json"
    local logfile="${OUTDIR}/${name}.log"

    # Skip if already fully complete
    if [[ -f "$outfile" ]]; then
        local n_tasks
        n_tasks=$(python3 -c "
import json
try:
    d = json.load(open('$outfile'))
    pq = d.get('per_question', [])
    errs = sum(1 for q in pq if any(p in (q.get('error','') or '').lower()
               for p in ['timeout','rate limit','overloaded','exhausted retries']))
    print(f'{len(pq)} {errs}')
except: print('0 0')
" 2>/dev/null)
        local total errs
        read -r total errs <<< "$n_tasks"
        if [[ "$total" -ge "$LIMIT" && "$errs" -eq 0 ]]; then
            echo "[SKIP] ${name}: already complete (${total} tasks)"
            return 0
        elif [[ "$total" -gt 0 ]]; then
            echo "[RESUME] ${name}: ${total} tasks found, ${errs} errors — resuming"
        fi
    fi

    echo ""
    echo "============================================================"
    echo "  Setting: ${name}"
    echo "  Model:   ${BACKBONE}"
    echo "  Bench:   ${BENCHMARK} (limit=${LIMIT})"
    echo "  Output:  ${outfile}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    if "${COMMON[@]}" "$@" --output "$outfile" 2>&1 | tee -a "$logfile"; then
        echo "[DONE] ${name} finished at $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "[FAIL] ${name} failed at $(date '+%Y-%m-%d %H:%M:%S') (exit=$?)"
    fi
}

# ==========================================================================
#  Setting 1: Thinking baseline (no tools, thinking ON)
#
#  Uses the model's native extended thinking (Claude/Gemini) or
#  "Think step by step" fallback (GPT-4o). No tool access.
#  deep_think mode: native extended thinking, no tools.
# ==========================================================================

run_exp "s1_thinking_baseline" \
    --tools off

# ==========================================================================
#  Setting 2: Domain tools — light (single-turn + scratchpad)
#
#  Single-turn function calling with domain-specific tools only
#  (clinical calculators, genomics, chemistry, etc.). No web search.
#  Thinking OFF by default for light mode.
#  light mode: domain tools + scratchpad context manager.
# ==========================================================================

CM_SCRATCHPAD=1 CM_SCRATCHPAD_MAX_TOKENS=16000 \
run_exp "s2_domain_light" \
    --tools biomed \
    --reasoning-mode light

# ==========================================================================
#  Setting 3: Domain tools — heavy (multi-turn ReAct + scratchpad)
#
#  Multi-turn ReAct loop with domain tools. The agent iteratively
#  decides which tools to call and can refine its answer.
#  Thinking ON by default for heavy mode.
#  heavy mode: domain tools + scratchpad context manager.
# ==========================================================================

CM_SCRATCHPAD=1 CM_SCRATCHPAD_MAX_TOKENS=16000 \
run_exp "s3_domain_heavy" \
    --tools biomed \
    --reasoning-mode heavy

# ==========================================================================
#  Setting 4: Web search — light (single-turn + scratchpad)
#
#  Single-turn function calling with web search tools only
#  (Serper Google search + Jina page reader). No domain tools.
#  Requires SERPER_API_KEY in .env.
#  light mode: web search tools + scratchpad context manager.
# ==========================================================================

CM_SCRATCHPAD=1 CM_SCRATCHPAD_MAX_TOKENS=16000 \
run_exp "s4_search_light" \
    --tools search \
    --reasoning-mode light

# ==========================================================================
#  Setting 5: Web search — heavy (multi-turn ReAct + scratchpad)
#
#  Multi-turn ReAct loop with web search tools only.
#  Requires SERPER_API_KEY in .env.
#  heavy mode: web search tools + scratchpad context manager.
# ==========================================================================

CM_SCRATCHPAD=1 CM_SCRATCHPAD_MAX_TOKENS=16000 \
run_exp "s5_search_heavy" \
    --tools search \
    --reasoning-mode heavy

# ==========================================================================
#  Setting 6: Combined (domain + web) — light (single-turn + scratchpad)
#
#  Single-turn function calling with both domain tools and web search.
#  Requires SERPER_API_KEY in .env.
#  light mode: all tools + scratchpad context manager.
# ==========================================================================

CM_SCRATCHPAD=1 CM_SCRATCHPAD_MAX_TOKENS=16000 \
run_exp "s6_combined_light" \
    --tools all \
    --reasoning-mode light

# ==========================================================================
#  Setting 7: Combined (domain + web) — heavy (multi-turn ReAct
#             + scratchpad + incremental summary)
#
#  Multi-turn ReAct loop with all tools and full context management
#  (scratchpad + incremental summary). This is the strongest setting.
#  Thinking ON by default for heavy mode.
#  Requires SERPER_API_KEY in .env.
#  heavy mode: all tools + scratchpad + incremental summary.
# ==========================================================================

CM_SCRATCHPAD=1 \
CM_SCRATCHPAD_MAX_TOKENS=16000 \
CM_INCREMENTAL_SUMMARY=1 \
CM_INCREMENTAL_THRESHOLD=160000 \
CM_INCREMENTAL_RECENT=50 \
CM_INCREMENTAL_BATCH=10 \
run_exp "s7_combined_heavy" \
    --tools all \
    --reasoning-mode heavy

# ==========================================================================
#  Summary
# ==========================================================================

echo ""
echo "=================================================================="
echo " BioMedArena Quick Run Complete"
echo " Model:     ${BACKBONE}"
echo " Benchmark: ${BENCHMARK}"
echo " Results:   ${OUTDIR}/"
echo " Finished:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="
echo ""
echo "Results:"
ls -lh "$OUTDIR"/s*.json 2>/dev/null || echo "  (no results yet)"
echo ""
echo "To compile results:  python compile_results.py"
echo "To view one result:  python -m json.tool ${OUTDIR}/s1_thinking_baseline.json | head -20"
