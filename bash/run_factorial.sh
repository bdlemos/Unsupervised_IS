#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Rate x selection factorial (Marcos' feedback, 2026-11)
# ══════════════════════════════════════════════════════════════════════════════
#
# Separates *how much* ESAE removes from *which* instances it removes:
#   rate      ∈ entropy (ESAE rule) | r20 (fixed 20%) | r21.3 (fixed = ESAE mean over the 19 datasets)
#   selection ∈ re (AE-weighted inside cells, = ESAE) | cells (uniform inside the same cells) | global (uniform, no cells)
# Every cell is repeated over SEEDS; a seed fixes tessellation, autoencoder init and sampling for every
# cell of that repetition, so cells are paired within (dataset, fold, seed).
# no-is (full data) is added once as the reference.
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
DATASETS="${DATASETS:-mpqa,trec,sst2,twitter,vader_movie,movie_review,sst1,pang_movie,subj,yelp_reviews,wos5736,reuters90,webkb,wos11967,acm,ohsumed,20ng,dblp,books}"
RATES="${RATES:-entropy r20 r21.3}"
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
