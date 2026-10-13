#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Overnight Extreme Rates Benchmark: ModernBERT on GPU
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs ModernBERT fine-tuning at severe data reduction rates (98% and 99%)
# comparing:
#   - typ-cover
#   - diff-cover
#   - sae-clean
#   - random
#
# Datasets: acm, books, dblp, webkb, 20ng
#
# All results and logs are strictly isolated in:
#   results/exp-extreme-rates/rate_98/
#   results/exp-extreme-rates/rate_99/
#
# Usage:
#   bash run_extreme_overnight.sh
#
# In tmux:
#   tmux new-session -s extreme-modernbert "bash run_extreme_overnight.sh"
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR
export DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/datasets}"

DATASETS="${DATASETS:-acm,books,dblp,webkb,20ng}"
RATES="${RATES:-98 99}"
BASE_METHODS="${BASE_METHODS:-typ-cover,diff-cover,sae-clean,random}"

MASTER_LOG="${ROOT_DIR}/results/exp-extreme-rates/overnight_execution.log"
mkdir -p "$(dirname "$MASTER_LOG")"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "═══════════════════════════════════════════════════════════════════"
echo "  ModernBERT Overnight Extreme Benchmark Started"
echo "  Date:     $(date)"
echo "  Datasets: ${DATASETS}"
echo "  Rates:    ${RATES}"
echo "  Methods:  ${BASE_METHODS}"
echo "  Log:      ${MASTER_LOG}"
echo "═══════════════════════════════════════════════════════════════════"

for r in $RATES; do
    echo ""
    echo "==================================================================="
    echo ">>> Starting Extreme Rate: ${r}% Reduction at $(date)"
    echo "==================================================================="
    
    RATE="$r" DATASETS="$DATASETS" BASE_METHODS="$BASE_METHODS" bash "${ROOT_DIR}/run_extreme_rates.sh"
    
    echo ">>> Finished Rate ${r}% at $(date)"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All Extreme Rates Completed Successfully at $(date)!"
echo "═══════════════════════════════════════════════════════════════════"
