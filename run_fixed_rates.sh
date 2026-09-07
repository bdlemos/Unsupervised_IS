#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Exp 4 — SAE-IS at Fixed Reduction Rates
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs SublinearAEIS (without entropy adaptation) at 7 fixed reduction rates:
#   10%, 15%, 20%, 25%, 30%, 35%, 40%
#
# For each rate, results go to:
#   results/exp4-fixed-rates/rate_<X>/
#
# Usage:
#   bash run_fixed_rates.sh
#   nohup bash run_fixed_rates.sh > exp4_fixed_rates.log 2>&1 &
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="sst2,ohsumed,reuters90,books,trec,20ng"
INPUT_REP="jina-v5"
BASE_RESULTS_DIR="${ROOT_DIR}/results/exp4-fixed-rates"

RATES="10 15 20 25 30 35 40"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Exp 4: SAE-IS at Fixed Reduction Rates"
echo "  Datasets:    ${DATASETS}"
echo "  Input Rep:   ${INPUT_REP}"
echo "  Rates:       ${RATES}"
echo "  Results Dir: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

for rate in ${RATES}; do
    RESULTS_DIR="${BASE_RESULTS_DIR}/rate_${rate}"

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Running SAE-IS at ${rate}% reduction → ${RESULTS_DIR}"
    echo "───────────────────────────────────────────────────────────────"

    RESULTS_DIR="${RESULTS_DIR}" bash "${ROOT_DIR}/run_pipeline.sh" \
        --methods "sae-rate-${rate}" \
        --datasets "${DATASETS}" \
        --inputrep "${INPUT_REP}" \
        --model "modernbert"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Exp 4: Fixed Rates COMPLETE"
echo "  Results in: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
