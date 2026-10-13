#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Extreme Rates Benchmark: ModernBERT at 98% or 99% Reduction
# ══════════════════════════════════════════════════════════════════════════════
#
# Compares Typ-Cover, Diffusion-Cover, SAE-Clean, and Random on ModernBERT
# under severe data reduction budgets (e.g. 98% or 99%).
#
# All results and logs live strictly isolated in:
#   results/exp-extreme-rates/rate_<RATE>/
#
# Usage:
#   bash run_extreme_rates.sh
#   RATE=98 DATASETS=acm bash run_extreme_rates.sh
#   RATE=99 DATASETS=20ng,books bash run_extreme_rates.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR
export DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/datasets}"

RATE="${RATE:-98}"
DATASETS="${DATASETS:-acm}"
BASE_METHODS="${BASE_METHODS:-typ-cover,diff-cover,sae-clean,random}"
BASE_RESULTS_DIR="${ROOT_DIR}/results/exp-extreme-rates"
RESULTS_DIR="${BASE_RESULTS_DIR}/rate_${RATE}"
export RESULTS_DIR

# Build method names with rate suffix (e.g. typ-cover-rate-98)
METHODS="$(python3 -c "
rate = '$RATE'
methods = '$BASE_METHODS'.split(',')
print(','.join([f'{m}-rate-{rate}' for m in methods]))
")"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Extreme Rates Experiment (ModernBERT on GPU)"
echo "  Rate:        ${RATE}% reduction"
echo "  Datasets:    ${DATASETS}"
echo "  Methods:     ${METHODS}"
echo "  Results Dir: ${RESULTS_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# ── Step 1: Generate selection splits ─────────────────────────────────────────
echo ""
echo ">>> [1/3] Gerando splits de seleção para ${DATASETS} em taxa ${RATE}%..."
python3 "${ROOT_DIR}/scripts/generate_extreme_splits.py" \
    --dataset "${DATASETS}" \
    --rate "${RATE}" \
    --methods "${BASE_METHODS}"

# ── Step 2: Run ModernBERT benchmark via atcBench ─────────────────────────────
echo ""
echo ">>> [2/3] Executando fine-tuning do ModernBERT em ${RESULTS_DIR}..."
cd "${ROOT_DIR}/atcBench"
bash run.sh --num-processes 1 --methods "${METHODS}" --datasets "${DATASETS}"

# ── Step 3: Aggregate results CSV ─────────────────────────────────────────────
echo ""
echo ">>> [3/3] Consolidando métricas em CSV..."
mkdir -p "${RESULTS_DIR}/classificacao/results"
python3 scripts/generate_results_csv.py \
    --pattern "${RESULTS_DIR}/classificacao/output/*/**/measures.fold_*.json" \
    --model "modernbert" \
    -o "${RESULTS_DIR}/classificacao/results/results_modernbert.csv"

cd "${ROOT_DIR}"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Experimento concluído!"
echo "  Resultados em: ${RESULTS_DIR}/classificacao/results/results_modernbert.csv"
echo "═══════════════════════════════════════════════════════════════════"
