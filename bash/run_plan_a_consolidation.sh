#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Plan 2026-11, block A: controlled consolidation, CURRENT regime, the 8 complete factorial datasets
# ══════════════════════════════════════════════════════════════════════════════
#
# Marcos' priorities 1 and 2: random, ESAE, biO-IS and full data with the same folds, seeds, regime
# and machine. No-IS, ESAE (fx-entropy-re) and random at the ESAE rate (fx-entropy-global) already
# exist in the factorial (results/exp-factorial, machine e6463628b158). This block adds the missing arms:
#   - biO-IS re-classified on its original split (1 run: its selection is fixed);
#   - random at biO-IS's per-fold count, seeds 1-3 (fx-matchbiois-global-seedN).
# Decision: IC/notes/decisions/2026-11-16_resposta-do-marcos.md
#
# Usage: bash bash/run_plan_a_consolidation.sh   (DRY_RUN=1 to check the biO-IS splits and print the plan)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/results}"
DATASETS="yelp_reviews,trec,webkb,mpqa,sst1,reuters90,ohsumed,20ng"

MODEL_CONFIG=modernbert \
SEEDS="1 2 3" \
INCLUDE_NO_IS=0 \
DATASETS="${DATASETS}" \
ANCHOR_DATASETS="${DATASETS}" \
RESULTS_DIR="${RESULTS_BASE}/exp-consolidation-current" \
    bash "${SCRIPT_DIR}/run_random_at_biois.sh"
