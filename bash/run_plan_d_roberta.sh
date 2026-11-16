#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Plan 2026-11, block D: RoBERTa, CURRENT regime, No-IS vs biO-IS vs random at biO-IS's rate
# ══════════════════════════════════════════════════════════════════════════════
#
# Marcos' priority 4 (a second classifier before E2SC/LSSM). All arms are new, on the same machine:
# the RoBERTa numbers from 2026-09 are NOT used (other code path and container).
# Datasets: reduction works (yelp_reviews, mpqa) and hurts (webkb, ohsumed), ascending size.
# Model: conf/model/roberta.yaml (current regime, patience 2 as ModernBERT); output under output/roberta/.
# Decision: IC/notes/decisions/2026-11-16_resposta-do-marcos.md
#
# Usage: bash bash/run_plan_d_roberta.sh   (DRY_RUN=1 to check the biO-IS splits and print the plan)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/results}"
DATASETS="yelp_reviews,mpqa,webkb,ohsumed"

MODEL_CONFIG=roberta \
SEEDS="1 2 3" \
INCLUDE_NO_IS=1 \
DATASETS="${DATASETS}" \
ANCHOR_DATASETS="${DATASETS}" \
RESULTS_DIR="${RESULTS_BASE}/exp-roberta-current" \
    bash "${SCRIPT_DIR}/run_random_at_biois.sh"
