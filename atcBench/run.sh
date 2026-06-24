#!/usr/bin/env bash

# run command -> nohup ./run.sh [--num-processes N] [--methods m1,m2,...] [--datasets d1,d2,...] > run.log 2>&1 &
# Exemplo com 4 processos: nohup ./run.sh --num-processes 4 > run.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Default values ─────────────────────────────────────────────────────────
DEFAULT_NUM_PROCESSES=1
DEFAULT_METHODS="adaptive-v2-perplexity,adaptive-perplexity"
DEFAULT_DATASETS="mpqa,reuters90,sst1,ohsumed,twitter,webkb,yelp_reviews,sst2,dblp,acm"

NUM_PROCESSES=$DEFAULT_NUM_PROCESSES
METHODS_ARG="$DEFAULT_METHODS"
DATASETS_ARG="$DEFAULT_DATASETS"

# ── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-processes|-n)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --methods)
            METHODS_ARG="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--num-processes <N>] [--methods <m1,m2,...>] [--datasets <d1,d2,...>]"
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to bash arrays
IFS=',' read -ra methods  <<< "$METHODS_ARG"
IFS=',' read -ra datasets <<< "$DATASETS_ARG"

echo "Running with $NUM_PROCESSES concurrent processes..."
echo "Methods  : ${methods[*]}"
echo "Datasets : ${datasets[*]}"

# Prefer project virtualenv python when available.
if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python"
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
export TB_CONFIG_NAME="config_default.yaml"
export PYTHON_BIN

# DICA: Para melhor balanceamento de carga, ordene os datasets do MAIOR para o MENOR.
# Assim, os mais pesados começam primeiro e os menores preenchem o tempo no final.

# Monta todas as combinações em um array
combos=()
for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    combos+=("${dataset}|${method}")
  done
done

# Usa xargs para gerenciar o pool de processos (workers)
printf "%s\n" "${combos[@]}" | xargs -n 1 -P "$NUM_PROCESSES" -I {} bash -c '
  IFS="|" read -r dataset method <<< "{}"
  combo="${dataset}_${method}"
  log_dir="${RESULTS_DIR:-/data/bernardolemos/results}/classificacao/logs/${dataset}"
  mkdir -p "$log_dir"
  log_file="$log_dir/${method}.log"
  
  echo "[RUN] Iniciando $combo (Acompanhe os detalhes em $log_file)"
  
  # Redireciona o stdout e stderr deste script especifico para o arquivo de log dele
  "$PYTHON_BIN" main.py "data=$combo" > "$log_file" 2>&1
  
  echo "[DONE] Finalizado $combo"
'

echo "All runs finished."