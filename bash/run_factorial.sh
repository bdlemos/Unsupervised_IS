#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Rate x selection factorial (Marcos' feedback, 2026-11)
# ══════════════════════════════════════════════════════════════════════════════
#
# Separates *how much* ESAE removes from *which* instances it removes:
#   rate      ∈ entropy (ESAE rule) | r20 (fixed 20%; the ESAE mean over the 19 datasets is 21.3%)
#   selection ∈ re (AE-weighted inside cells, = ESAE) | cells (uniform inside the same cells) | global (uniform, no cells)
# Every cell is repeated over SEEDS; a seed fixes tessellation, autoencoder init and sampling for every
# cell of that repetition, so cells are paired within (dataset, fold, seed).
# no-is (full data) is added once as the reference.
#
# Default datasets (chosen before any factorial result, from earlier ModernBERT runs only):
#   - ESAE rate far from 20%, so the rate factor can matter: ohsumed 12.0, trec 14.0, books 18.1 | sst1 24.8, yelp_reviews 27.4, mpqa 28.8
#   - reduction-sensitive / failed non-inferiority vs No-IS: books, ohsumed, sst1, webkb
#   - known adverse cases that must be reported: webkb (minority-class loss), 20ng (ESAE < random at same rate)
#   - all exp2 ablation datasets except sst2, for continuity: 20ng, books, ohsumed, reuters90, trec
#   - extreme imbalance: reuters90, ohsumed, webkb
#   - one per source group (sst1 not sst2; no wos / movie duplicates); dblp and acm left out (rate ≈ 20%, slowest)
#
# Method names: fx-<rate>-<selection>-seed<N>  (see unsupervised-is/src/iSel/factorial_is.py)
# Each selection writes split_<folds>_<method>_params.json with the per-fold decision values.
#
# Usage:
#   bash bash/run_factorial.sh
#   DATASETS=books,webkb SEEDS="1 2" STEPS=1,2 bash bash/run_factorial.sh   # selection only, quick check
#   DRY_RUN=1 bash bash/run_factorial.sh                                    # print the plan and exit
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASETS="${DATASETS:-ohsumed,trec,books,20ng,webkb,reuters90,sst1,mpqa,yelp_reviews}"
RATES="${RATES:-entropy r20}"
SELECTIONS="${SELECTIONS:-re cells global}"
SEEDS="${SEEDS:-1 2 3 4 5}"
INPUT_REP="${INPUT_REP:-jina-v5}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/exp-factorial}"

methods=("no-is")
for seed in ${SEEDS}; do
    for rate in ${RATES}; do
        for selection in ${SELECTIONS}; do
            methods+=("fx-${rate}-${selection}-seed${seed}")
        done
    done
done
METHODS="$(IFS=,; echo "${methods[*]}")"

n_datasets="$(awk -F, '{print NF}' <<< "${DATASETS}")"
n_methods="${#methods[@]}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Rate x selection factorial"
echo "  Datasets:    ${DATASETS}"
echo "  Rates:       ${RATES}"
echo "  Selections:  ${SELECTIONS}"
echo "  Seeds:       ${SEEDS}"
echo "  Methods:     ${n_methods} (incl. no-is)"
echo "  Selections × folds to classify: ${n_datasets} datasets × ${n_methods} methods × 10 folds = $((n_datasets * n_methods * 10)) fine-tunings (agnews has 5 folds)"
echo "  Results Dir: ${RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%s\n' "${methods[@]}"
    exit 0
fi

PIPELINE_ARGS=(
    --methods "${METHODS}"
    --datasets "${DATASETS}"
    --inputrep "${INPUT_REP}"
    --model "modernbert"
)
if [[ -n "${STEPS:-}" ]]; then
    PIPELINE_ARGS+=(--steps "${STEPS}")
fi

# Records which code produced the splits (the Docker image has no .git)
GIT_COMMIT="${GIT_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)}"
export GIT_COMMIT

RESULTS_DIR="${RESULTS_DIR}" bash "${SCRIPT_DIR}/run_pipeline.sh" "${PIPELINE_ARGS[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Factorial COMPLETE — results in ${RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
