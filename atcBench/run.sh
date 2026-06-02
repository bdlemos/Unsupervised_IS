#!/usr/bin/env bash

# run command -> nohup ./run.sh [NUM_PROCESSES] > run.log 2>&1 &
# Exemplo com 4 processos: nohup ./run.sh 4 > run.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

NUM_PROCESSES=${1:-1} # Padrão para 1 processo caso não seja informado

echo "Running with $NUM_PROCESSES concurrent processes..."

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
datasets=(mpqa reuters90 sst1 ohsumed twitter webkb yelp_reviews sst2 dblp acm )
methods=(adaptive-v2-perplexity adaptive-perplexity)
# methods=(autoencoder-is random-is biois gmm-is perplexity-is no-is)

# Monta todas as combinações em um array
combos=()
for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    combos+=("${dataset}_${method}")
  done
done

# Cria uma pasta para separar os logs e não bagunçar
mkdir -p logs

# Usa xargs para gerenciar o pool de processos (workers)
printf "%s\n" "${combos[@]}" | xargs -n 1 -P "$NUM_PROCESSES" -I {} bash -c '
  combo="{}"
  log_file="logs/run_${combo}.log"
  echo "[RUN] Iniciando $combo (Acompanhe os detalhes em $log_file)"
  
  # Redireciona o stdout e stderr deste script especifico para o arquivo de log dele
  "$PYTHON_BIN" main.py "data=$combo" > "$log_file" 2>&1
  
  echo "[DONE] Finalizado $combo"
'

echo "All runs finished."