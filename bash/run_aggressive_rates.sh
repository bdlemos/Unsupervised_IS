#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Aggressive Fixed Rates Comparison: SAE-IS vs Random IS
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs both SAE-IS and Random IS at aggressive reduction rates (e.g. 40%, 60%)
# saving both methods under the exact same directory structure for easy comparison:
#   results/exp-aggressive-rates/rate_<X>/
#
# Usage:
#   bash bash/run_aggressive_rates.sh
#   nohup bash bash/run_aggressive_rates.sh > aggressive_rates.log 2>&1 &
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASETS="${DATASETS:-books,dblp,webkb}"
INPUT_REP="${INPUT_REP:-jina-v5}"
BASE_RESULTS_DIR="${ROOT_DIR}/results/exp-aggressive-rates"
RATES="${RATES:-40 60}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Aggressive Rates Experiment: SAE-IS vs Random IS vs Softmax SAE"
echo "  Datasets:    ${DATASETS}"
echo "  Input Rep:   ${INPUT_REP}"
echo "  Rates:       ${RATES}"
echo "  Results Dir: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

for rate in ${RATES}; do
    RESULTS_DIR="${BASE_RESULTS_DIR}/rate_${rate}"

    if [[ -n "${METHODS:-}" ]]; then
        RATE_METHODS="${METHODS//\{rate\}/$rate}"
    else
        RATE_METHODS="sae-softmax-rate-${rate},random-rate-${rate}"
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Running methods [${RATE_METHODS}] at ${rate}% reduction"
    echo "  Target dir: ${RESULTS_DIR}"
    echo "───────────────────────────────────────────────────────────────"

    PIPELINE_ARGS=(
        --methods "${RATE_METHODS}"
        --datasets "${DATASETS}"
        --inputrep "${INPUT_REP}"
        --model "modernbert"
    )
    if [[ -n "${STEPS:-}" ]]; then
        PIPELINE_ARGS+=(--steps "${STEPS}")
    fi

    # Runs methods together into the same directory for direct comparison
    RESULTS_DIR="${RESULTS_DIR}" bash "${SCRIPT_DIR}/run_pipeline.sh" "${PIPELINE_ARGS[@]}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Aggressive Rates Experiment COMPLETE"
echo "  Results saved in: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"