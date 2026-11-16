#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Plan 2026-11, block B: F1 x rate curves, CURRENT regime, random vs RE at common fixed rates
# ══════════════════════════════════════════════════════════════════════════════
#
# Marcos' priority 3: the direct comparison across rates that "how much to remove" needs.
# Rates 10/40/60% here; the 20% point and No-IS come from the factorial (fx-r20-{re,global}, no-is).
# Datasets: reduction works (No-IS non-inferior in 6/5/3 of 6 factorial arms) yelp_reviews, mpqa, trec;
# reduction hurts (0/6) webkb, reuters90, ohsumed. Ascending size, so small ones finish first.
# Pre-specified analysis: per dataset (NB + Holm), dose = F1(10%) - F1(60%) of random, and
# RE - random at each rate; "how much" needs dose significant and larger than |RE - random|.
# Decision: IC/notes/decisions/2026-11-16_resposta-do-marcos.md
#
# Usage: bash bash/run_plan_b_rate_curves.sh   (DRY_RUN=1 to print the plan)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/results}"

export MODEL_CONFIG=modernbert
DATASETS="yelp_reviews,trec,webkb,mpqa,reuters90,ohsumed" \
RATES="r10 r40 r60" \
SELECTIONS="re global" \
SEEDS="1 2 3" \
INCLUDE_NO_IS=0 \
RESULTS_DIR="${RESULTS_BASE}/exp-rate-curves" \
    bash "${SCRIPT_DIR}/run_factorial.sh"
