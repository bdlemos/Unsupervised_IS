#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Exp 3 — Sensitivity Analysis of ESAE-IS Parameters
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs ESAE-IS with one-at-a-time parameter variations, keeping defaults
# for all other parameters.
#
# Usage:
#   bash run_sensitivity.sh
#   nohup bash run_sensitivity.sh > exp3_sensitivity.log 2>&1 &
#
# Each configuration produces its own results directory under:
#   results/exp3-sensitivity/<param>_<value>/
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="sst2,ohsumed,reuters90,books,trec,20ng"
INPUT_REP="jina-v5"
BASE_RESULTS_DIR="${ROOT_DIR}/results/exp3-sensitivity"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Exp 3: Sensitivity Analysis"
echo "  Datasets:    ${DATASETS}"
echo "  Input Rep:   ${INPUT_REP}"
echo "  Results Dir: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# ── Helper: run one sensitivity configuration ─────────────────────────────
run_config() {
    local method_name="$1"
    local results_subdir="$2"
    local results_dir="${BASE_RESULTS_DIR}/${results_subdir}"

    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  Running: ${method_name} → ${results_dir}"
    echo "───────────────────────────────────────────────────────────────"

    RESULTS_DIR="${results_dir}" bash "${ROOT_DIR}/run_pipeline.sh" \
        --methods "${method_name}" \
        --datasets "${DATASETS}" \
        --inputrep "${INPUT_REP}" \
        --model "modernbert"
}

# ══════════════════════════════════════════════════════════════════════════════
# K (n_clusters) variations — default is sqrt(N)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "▶ Varying K (n_clusters): 50, 100, 200, sqrt(N)"

run_config "esae-K50"     "K_50"
run_config "esae-K100"    "K_100"
run_config "esae-K200"    "K_200"
run_config "esae-KsqrtN"  "K_sqrtN"  # default

# ══════════════════════════════════════════════════════════════════════════════
# α (alpha) variations — default is 15
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "▶ Varying α (alpha): 5, 10, 15, 20, 25"

run_config "esae-alpha5"  "alpha_5"
run_config "esae-alpha10" "alpha_10"
# alpha=15 + sqrtN is the default, already covered above as K_sqrtN
# But we run it again to have it in the alpha_ directory for clean analysis
run_config "esae-alpha15" "alpha_15"
run_config "esae-alpha20" "alpha_20"
run_config "esae-alpha25" "alpha_25"

# ══════════════════════════════════════════════════════════════════════════════
# γ (gamma) variations — default is 0.5
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "▶ Varying γ (gamma): 0.3, 0.5, 0.7"

run_config "esae-gamma03" "gamma_03"
# gamma=0.5 is the default (covered above)
run_config "esae-gamma05" "gamma_05"
run_config "esae-gamma07" "gamma_07"

# ══════════════════════════════════════════════════════════════════════════════
# r_max variations — default is 0.5
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "▶ Varying r_max: 0.3, 0.4, 0.5, 0.6"

run_config "esae-rmax03" "rmax_03"
run_config "esae-rmax04" "rmax_04"
# rmax=0.5 is the default (covered above)
run_config "esae-rmax05" "rmax_05"
run_config "esae-rmax06" "rmax_06"

# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Exp 3: Sensitivity Analysis COMPLETE"
echo "  Results in: ${BASE_RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
