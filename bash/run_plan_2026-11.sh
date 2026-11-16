#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Plan 2026-11: every block in one GPU queue, back to back, so the GPU is never left idle
# ══════════════════════════════════════════════════════════════════════════════
#
# Blocks (each also runs alone; see its script):
#   A  run_plan_a_consolidation.sh   current regime, 8 datasets: biO-IS + random at biO-IS rate   ~26 h
#   B  run_plan_b_rate_curves.sh     current regime, 6 datasets: RE and random at 10/40/60%      ~79 h
#   C  run_plan_c_legacy_anchor.sh   legacy regime, 19 datasets: biO-IS + No-IS, 1 run each      ~41 h
#   D  run_plan_d_roberta.sh         RoBERTa, current regime, 4 datasets                         ~20-30 h
# Decision and estimates: IC/notes/decisions/2026-11-16_resposta-do-marcos.md
#
# What this script adds over running the blocks by hand:
#   - preflight before any GPU hour: machine, GPU, datasets, biO-IS splits and every block's plan
#     (DRY_RUN of each block), so a missing file fails now and not 60 h in;
#   - blocks run strictly one after another (no overlap: concurrent jobs would distort time_train,
#     which the cost analysis uses); a failed block does not stop the queue, the next one starts;
#   - failed blocks are retried once at the end (finished folds are skipped, so a retry only redoes
#     what is missing, e.g. after a transient OOM);
#   - provenance (machine, GPU, driver, library versions, commit) and a status line per block.
#
# The paired comparisons only hold on the machine of the existing runs (e6463628b158, the container
# hostname recorded in every measures.json). The script refuses another hostname unless
# ALLOW_OTHER_MACHINE=1; if the machine changes, blocks A and C have to be redone together.
#
# Usage (from the repo root, inside the GPU container):
#   DRY_RUN=1 bash bash/run_plan_2026-11.sh                    # preflight only
#   nohup bash bash/run_plan_2026-11.sh > plan_2026-11.log 2>&1 &
#   BLOCKS="B D" bash bash/run_plan_2026-11.sh                 # a subset, in this order
#   tail -f results/plan-2026-11/status.tsv                     # progress
# ══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ROOT_DIR
export RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/results}"
export DATASETS_DIR="${DATASETS_DIR:-${ROOT_DIR}/datasets}"
export MATCH_SELECTION_DIR="${MATCH_SELECTION_DIR:-${ROOT_DIR}/results/jina-results/jina-v5/instance_selection/selection}"
BLOCKS="${BLOCKS:-A B C D}"
EXPECTED_MACHINE="${EXPECTED_MACHINE:-e6463628b158}"
ALLOW_OTHER_MACHINE="${ALLOW_OTHER_MACHINE:-0}"
RETRY_FAILED="${RETRY_FAILED:-1}"
PLAN_DIR="${RESULTS_BASE}/plan-2026-11"
LOG_DIR="${PLAN_DIR}/logs"
STATUS_FILE="${PLAN_DIR}/status.tsv"
# Model config and selection seeds are set by each block; never inherit them from the caller
unset MODEL_CONFIG SEEDS INCLUDE_NO_IS ANCHOR_DATASETS DATASETS RATES SELECTIONS RESULTS_DIR STEPS

declare -A BLOCK_SCRIPT=(
    [A]=run_plan_a_consolidation.sh
    [B]=run_plan_b_rate_curves.sh
    [C]=run_plan_c_legacy_anchor.sh
    [D]=run_plan_d_roberta.sh
)
# Every dataset any block touches (C covers the 19)
ALL_DATASETS="mpqa trec sst2 twitter vader_movie movie_review sst1 pang_movie subj yelp_reviews wos5736 reuters90 webkb wos11967 ohsumed 20ng acm dblp books"

if [[ -x "/app/venv/bin/python" ]]; then
    PYTHON_BIN="/app/venv/bin/python"
elif [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*"; }

# ── Preflight ─────────────────────────────────────────────────────────────────
say "Plan 2026-11 — blocks: ${BLOCKS}"
errors=()

for block in ${BLOCKS}; do
    [[ -n "${BLOCK_SCRIPT[${block}]:-}" ]] || errors+=("unknown block '${block}' (use A B C D)")
done

machine="$(hostname)"
if [[ "${machine}" != "${EXPECTED_MACHINE}" ]]; then
    if [[ "${ALLOW_OTHER_MACHINE}" == "1" ]]; then
        say "WARNING: machine ${machine} != ${EXPECTED_MACHINE}; pairing with the existing runs is lost"
    else
        errors+=("machine is ${machine}, expected ${EXPECTED_MACHINE} (ALLOW_OTHER_MACHINE=1 to override; then redo A and C together)")
    fi
fi

if ! gpu_info="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)"; then
    errors+=("no GPU visible: ${gpu_info}")
fi

for dataset in ${ALL_DATASETS}; do
    [[ -f "${DATASETS_DIR}/${dataset}/texts.txt" ]] || errors+=("missing ${DATASETS_DIR}/${dataset}/texts.txt")
done

mkdir -p "${LOG_DIR}"
for block in ${BLOCKS}; do
    script="${BLOCK_SCRIPT[${block}]:-}"
    [[ -n "${script}" ]] || continue
    if ! DRY_RUN=1 bash "${SCRIPT_DIR}/${script}" > "${LOG_DIR}/preflight_${block}.log" 2>&1; then
        errors+=("block ${block} preflight failed, see ${LOG_DIR}/preflight_${block}.log")
    fi
done

if [[ "${#errors[@]}" -gt 0 ]]; then
    say "PREFLIGHT FAILED:"
    printf '  - %s\n' "${errors[@]}"
    exit 1
fi
say "Preflight ok (machine ${machine}; GPU ${gpu_info}); block plans in ${LOG_DIR}/preflight_*.log"

if [[ -n "${DRY_RUN:-}" ]]; then
    for block in ${BLOCKS}; do
        echo "──── block ${block}: ${BLOCK_SCRIPT[${block}]}"
        cat "${LOG_DIR}/preflight_${block}.log"
    done
    exit 0
fi

# ── Provenance ────────────────────────────────────────────────────────────────
export GIT_COMMIT="${GIT_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)}"
{
    echo "started_at: $(stamp)"
    echo "machine: ${machine}"
    echo "gpu: ${gpu_info}"
    echo "git_commit: ${GIT_COMMIT}"
    echo "blocks: ${BLOCKS}"
    "${PYTHON_BIN}" -c "import sys, torch, transformers; print('python:', sys.version.split()[0]); print('torch:', torch.__version__, '| cuda:', torch.version.cuda); print('transformers:', transformers.__version__)" 2>&1
} > "${PLAN_DIR}/provenance_$(date +%Y%m%dT%H%M%S).txt"
[[ -f "${STATUS_FILE}" ]] || printf 'block\tattempt\tstarted\tfinished\texit\tlog\n' > "${STATUS_FILE}"

# ── Queue ─────────────────────────────────────────────────────────────────────
run_block() {
    local block="$1" attempt="$2"
    local log="${LOG_DIR}/block_${block}.log"
    local started
    started="$(stamp)"
    say "▶ block ${block} (attempt ${attempt}): ${BLOCK_SCRIPT[${block}]} — log ${log}"
    echo "===== $(stamp) attempt ${attempt} =====" >> "${log}"
    bash "${SCRIPT_DIR}/${BLOCK_SCRIPT[${block}]}" >> "${log}" 2>&1
    local code=$?
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${block}" "${attempt}" "${started}" "$(stamp)" "${code}" "${log}" >> "${STATUS_FILE}"
    if [[ "${code}" -eq 0 ]]; then
        say "✔ block ${block} done"
    else
        say "✘ block ${block} failed (exit ${code}); moving on, see ${log}"
    fi
    return "${code}"
}

failed=()
for block in ${BLOCKS}; do
    run_block "${block}" 1 || failed+=("${block}")
done

if [[ "${#failed[@]}" -gt 0 && "${RETRY_FAILED}" == "1" ]]; then
    say "Retrying failed blocks once: ${failed[*]}"
    still_failed=()
    for block in "${failed[@]}"; do
        run_block "${block}" 2 || still_failed+=("${block}")
    done
    failed=(${still_failed[@]+"${still_failed[@]}"})
fi

if [[ "${#failed[@]}" -gt 0 ]]; then
    say "Plan finished with failed blocks: ${failed[*]} (status in ${STATUS_FILE})"
    exit 1
fi
say "Plan finished: all blocks done (status in ${STATUS_FILE})"
