#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Plan 2026-11, block C: LEGACY-regime anchor for the 19-dataset random-at-biO-IS table
# ══════════════════════════════════════════════════════════════════════════════
#
# Random at biO-IS's rate already exists for the 19 datasets (legacy regime, 3 seeds, machine
# e6463628b158), but biO-IS and No-IS are single runs from 2026-09 in other containers. This block
# re-runs both, once each, under the legacy regime on the same machine, so that table is controlled.
# No random arm here (SEEDS=""), the existing one is reused.
# Decision: IC/notes/decisions/2026-11-16_resposta-do-marcos.md
#
# Usage: bash bash/run_plan_c_legacy_anchor.sh   (DRY_RUN=1 to check the biO-IS splits and print the plan)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/results}"
# The 19 datasets of paper/gen_main_table.py, roughly ascending size
DATASETS="mpqa,trec,sst2,twitter,vader_movie,movie_review,sst1,pang_movie,subj,yelp_reviews,wos5736,reuters90,webkb,wos11967,ohsumed,20ng,acm,dblp,books"

MODEL_CONFIG=modernbert_legacy \
SEEDS="" \
INCLUDE_NO_IS=1 \
DATASETS="${DATASETS}" \
ANCHOR_DATASETS="${DATASETS}" \
RESULTS_DIR="${RESULTS_BASE}/exp-legacy-anchor" \
    bash "${SCRIPT_DIR}/run_random_at_biois.sh"
