#!/usr/bin/env bash
# run command -> nohup ./bash/run_unsupervised_selection.sh > run.log 2>&1 &
# bash bash/run_unsupervised_selection.sh [--methods m1,m2,...] [--datasets d1,d2,...]

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# ── Default values ─────────────────────────────────────────────────────────
DEFAULT_METHODS="adaptive-v2-perplexity,adaptive-perplexity"
DEFAULT_DATASETS="mpqa,reuters90,sst1,ohsumed,twitter,webkb,yelp_reviews,sst2,dblp,acm"
DEFAULT_INPUT_REP="tfidf"

METHODS_ARG="$DEFAULT_METHODS"
DATASETS_ARG="$DEFAULT_DATASETS"
INPUT_REP_ARG="$DEFAULT_INPUT_REP"


# ── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)
            METHODS_ARG="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_ARG="$2"
            shift 2
            ;;
        --inputrep)
            INPUT_REP_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--methods <m1,m2,...>] [--datasets <d1,d2,...>] [--inputrep <input_rep>]"
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to bash arrays
IFS=',' read -ra methods  <<< "$METHODS_ARG"
IFS=',' read -ra datasets <<< "$DATASETS_ARG"

datain="${DATASETS_DIR:-$WORKDIR/../datasets}"
out="${RESULTS_DIR:-$WORKDIR/../results}/instance_selection"

mkdir -p "$out"
mkdir -p "$out/logs"

echo "Methods  : ${methods[*]}"
echo "Datasets : ${datasets[*]}"

for dataset in "${datasets[@]}"; do
    echo "Dataset: $dataset"
    for method in "${methods[@]}"; do
        echo "  Iniciando $dataset e $method (Log: $out/logs/${dataset}_${method}.log)"
        python scripts/run_generateSplit.py -d "$dataset" -m "$method" --datain "$datain" --out "$out" --inputrep "$INPUT_REP_ARG" > "$out/logs/${dataset}_${method}.log" 2>&1
    done
done
