#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Random selection at biO-IS's per-fold reduction, LEGACY ModernBERT training (2026-11)
# ══════════════════════════════════════════════════════════════════════════════
#
# Does the supervised biO-IS beat plain random removal of the same number of instances?
# For every (dataset, fold, seed) the random selection keeps exactly as many training instances as
# biO-IS kept in that fold, chosen uniformly (fx-matchbiois-global, see FactorialIS rate="matched"
# and run_generateSplit.reference_keep_counts). Classification uses the LEGACY training regime
# (conf/model/modernbert_legacy.yaml: Adam without effective weight decay, cumulative val-loss
# patience, last-epoch weights), the regime of the biO-IS and No-IS numbers in the main table
# (experiments/2026-09_main-results-jina-v5), so each random fold pairs with the biO-IS fold.
#
# biO-IS per-fold kept counts are read from its selection splits:
#   ${MATCH_SELECTION_DIR}/<dataset>/split_10_biois.pkl
# default MATCH_SELECTION_DIR = results/jina-results/jina-v5/instance_selection/selection
# (the splits the main-table biO-IS was classified on; gitignored, copy them to the GPU machine).
#
# Method names: fx-matchbiois-<selection>-seed<N>; each selection writes
# split_10_<method>_params.json with match.target_n_kept / target_n_removed per fold.
#
# Usage:
#   DRY_RUN=1 bash bash/run_random_at_biois.sh               # print the plan, check the biO-IS splits, exit
#   SEEDS=1 bash bash/run_random_at_biois.sh                 # one repetition first
#   bash bash/run_random_at_biois.sh                         # seeds 1 2 3 (completed work is skipped)
#   ANCHOR_DATASETS=trec,webkb bash bash/run_random_at_biois.sh   # also re-classify biO-IS itself under the
#                                                            # legacy regime, to check it reproduces the table
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# The 19 datasets of paper/gen_main_table.py, roughly ascending size (small ones finish first)
DATASETS="${DATASETS:-mpqa,trec,sst2,twitter,vader_movie,movie_review,sst1,pang_movie,subj,yelp_reviews,wos5736,reuters90,webkb,wos11967,ohsumed,20ng,acm,dblp,books}"
SEEDS="${SEEDS:-1 2 3}"
SELECTIONS="${SELECTIONS:-global}"
REFERENCE="biois"
INPUT_REP="${INPUT_REP:-jina-v5}"
FOLDS=10
ANCHOR_DATASETS="${ANCHOR_DATASETS:-}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/exp-random-at-biois}"
MATCH_SELECTION_DIR="${MATCH_SELECTION_DIR:-${ROOT_DIR}/results/jina-results/jina-v5/instance_selection/selection}"
export MODEL_CONFIG="${MODEL_CONFIG:-modernbert_legacy}"
export MATCH_SELECTION_DIR
export DATASETS_DIR="${DATASETS_DIR:-${ROOT_DIR}/datasets}"
if [[ -d "/app/venv" ]]; then
    export VENV_PATH="/app/venv"
else
    export VENV_PATH="${VENV_PATH:-${ROOT_DIR}/venv}"
fi

methods=()
for seed in ${SEEDS}; do
    for selection in ${SELECTIONS}; do
        methods+=("fx-match${REFERENCE}-${selection}-seed${seed}")
    done
done
METHODS="$(IFS=,; echo "${methods[*]}")"

IFS=',' read -ra dataset_list <<< "${DATASETS}"
IFS=',' read -ra anchor_list <<< "${ANCHOR_DATASETS}"
n_datasets="${#dataset_list[@]}"
n_methods="${#methods[@]}"
n_anchor="${#anchor_list[@]}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Random at biO-IS rate (legacy training)"
echo "  Datasets:    ${DATASETS}"
echo "  Seeds:       ${SEEDS}"
echo "  Rule:        ${SELECTIONS}"
echo "  Model:       conf/model/${MODEL_CONFIG}.yaml"
echo "  biO-IS from: ${MATCH_SELECTION_DIR}/<dataset>/split_${FOLDS}_${REFERENCE}.pkl"
echo "  Selections:  ${n_datasets} datasets × ${n_methods} methods = $((n_datasets * n_methods)) runs ($((n_datasets * n_methods * FOLDS)) fold selections)"
echo "  Fine-tunings: ${n_datasets} × ${n_methods} × ${FOLDS} folds = $((n_datasets * n_methods * FOLDS))"
if [[ "${n_anchor}" -gt 0 ]]; then
    echo "  Anchor:      biO-IS re-classified on ${ANCHOR_DATASETS}: $((n_anchor * FOLDS)) more fine-tunings"
fi
echo "  Results Dir: ${RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# Every dataset needs its biO-IS split before anything runs
missing=()
for dataset in "${dataset_list[@]}" "${anchor_list[@]}"; do
    [[ -z "${dataset}" ]] && continue
    reference_split="${MATCH_SELECTION_DIR}/${dataset}/split_${FOLDS}_${REFERENCE}.pkl"
    if [[ -f "${reference_split}" ]]; then
        reduction="$(python3 -c "import json,sys; i=json.load(open(sys.argv[1]))['info']; print(f\"{sum(1-k/n for k,n in zip(i['reduced_len'],i['original_len']))/len(i['reduced_len']):.2%}\")" \
            "${MATCH_SELECTION_DIR}/${dataset}/saida_${REFERENCE}.json" 2>/dev/null || echo "?")"
        echo "  [ok] ${dataset}: biO-IS mean reduction ${reduction}"
    else
        missing+=("${reference_split}")
    fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "ERROR: biO-IS selection splits not found (set MATCH_SELECTION_DIR to the folder holding <dataset>/split_${FOLDS}_${REFERENCE}.pkl):"
    printf '  %s\n' "${missing[@]}"
    exit 1
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%s\n' "${methods[@]}"
    exit 0
fi

# Records which code produced the splits (the Docker image has no .git)
GIT_COMMIT="${GIT_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)}"
export GIT_COMMIT

PIPELINE_ARGS=(
    --methods "${METHODS}"
    --datasets "${DATASETS}"
    --inputrep "${INPUT_REP}"
    --model "modernbert"
)
if [[ -n "${STEPS:-}" ]]; then
    PIPELINE_ARGS+=(--steps "${STEPS}")
fi

# Anchor first: biO-IS itself under the legacy regime, on its original split (copied, not re-selected),
# so a mismatch with the main table shows up before the long run (and lands in the pipeline's CSVs)
if [[ "${n_anchor}" -gt 0 && ( -z "${STEPS:-}" || ",${STEPS}," =~ ,(3|benchmark), ) ]]; then
    for dataset in "${anchor_list[@]}"; do
        target="${RESULTS_DIR}/instance_selection/selection/${dataset}"
        mkdir -p "${target}"
        cp -n "${MATCH_SELECTION_DIR}/${dataset}/split_${FOLDS}_${REFERENCE}.pkl" "${target}/"
    done
    (cd "${ROOT_DIR}/atcBench" && RESULTS_DIR="${RESULTS_DIR}" bash run.sh --num-processes 1 \
        --methods "${REFERENCE}" --datasets "${ANCHOR_DATASETS}" --model-config "${MODEL_CONFIG}")
fi

RESULTS_DIR="${RESULTS_DIR}" bash "${SCRIPT_DIR}/run_pipeline.sh" "${PIPELINE_ARGS[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Random at biO-IS rate COMPLETE — results in ${RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"
