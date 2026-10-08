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
#   bash run_aggressive_rates.sh
#   nohup bash run_aggressive_rates.sh > aggressive_rates.log 2>&1 &
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="webkb,dblp,books,agnews,acm,20ng,sst1"
INPUT_REP="jina-v5"
BASE_RESULTS_DIR="${ROOT_DIR}/results/exp-aggressive-rates"
RATES="40 60"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Aggressive Rates Experiment: SAE-IS vs Random IS"
echo "  Datasets:    ${DATASETS}"
echo "  Input Rep:   ${INPUT_REP}"
echo "  Rates:       ${RATES}"
echo "  Results Dir: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

for rate in ${RATES}; do
    RESULTS_DIR="${BASE_RESULTS_DIR}/rate_${rate}"

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Running SAE-IS and Random IS at ${rate}% reduction"
    echo "  Target dir: ${RESULTS_DIR}"
    echo "───────────────────────────────────────────────────────────────"

    # Runs both methods together into the same directory for direct comparison
    RESULTS_DIR="${RESULTS_DIR}" bash "${ROOT_DIR}/run_pipeline.sh" \
        --methods "sae-rate-${rate},random-rate-${rate}" \
        --datasets "${DATASETS}" \
        --inputrep "${INPUT_REP}" \
        --model "modernbert"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Aggressive Rates Experiment COMPLETE"
echo "  Results saved in: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
