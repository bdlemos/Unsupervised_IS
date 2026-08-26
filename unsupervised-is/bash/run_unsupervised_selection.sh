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

# Convert comma-separated strings to bash arrays, trimming whitespace from each element
IFS=',' read -ra methods_raw  <<< "$METHODS_ARG"
IFS=',' read -ra datasets_raw <<< "$DATASETS_ARG"

methods=()
for m in "${methods_raw[@]}"; do
    methods+=("$(echo "$m" | xargs)")
done

datasets=()
for d in "${datasets_raw[@]}"; do
    datasets+=("$(echo "$d" | xargs)")
done

datain="${DATASETS_DIR:-$WORKDIR/../datasets}"
out="${RESULTS_DIR:-$WORKDIR/../results}/instance_selection"

mkdir -p "$out"
mkdir -p "$out/logs"

echo "Methods  : ${methods[*]}"
echo "Datasets : ${datasets[*]}"

# ── Error tracking ──────────────────────────────────────────────────────────
FAILED_RUNS=()
FAILED_PAIRS_FILE="$out/logs/failed_pairs.csv"
FAILED_METHODS_FILE="$out/logs/failed_methods.txt"

# Limpa arquivos de falha de execuções anteriores antes de começar
> "$FAILED_PAIRS_FILE"
> "$FAILED_METHODS_FILE"

for dataset in "${datasets[@]}"; do
    echo "Dataset: $dataset"
    for method in "${methods[@]}"; do
        echo "  Iniciando $dataset e $method (Log: $out/logs/${dataset}_${method}.log)"

        set +e
        python scripts/run_generateSplit.py -d "$dataset" -m "$method" --datain "$datain" --out "$out" --inputrep "$INPUT_REP_ARG" > "$out/logs/${dataset}_${method}.log" 2>&1
        exit_code=$?
        set -e

        if [[ $exit_code -ne 0 ]]; then
            echo "  ERROR: $dataset / $method failed (exit code $exit_code). See $out/logs/${dataset}_${method}.log"
            FAILED_RUNS+=("$dataset,$method")
            echo "$dataset,$method" >> "$FAILED_PAIRS_FILE"
        fi
    done
done

# ── Deriva a lista de métodos que falharam em TODOS os datasets ────────────
# (só removemos um método da lista global se ele falhou sempre — se falhou
#  só em alguns datasets, mantemos, pois pode ser um problema pontual daquele
#  dataset e não do método em si)
if [[ -s "$FAILED_PAIRS_FILE" ]]; then
    for method in "${methods[@]}"; do
        total_for_method=0
        failed_for_method=0
        for dataset in "${datasets[@]}"; do
            total_for_method=$((total_for_method + 1))
            if grep -qx "${dataset},${method}" "$FAILED_PAIRS_FILE"; then
                failed_for_method=$((failed_for_method + 1))
            fi
        done
        if [[ $failed_for_method -eq $total_for_method ]]; then
            echo "$method" >> "$FAILED_METHODS_FILE"
        fi
    done
fi

# ── Final report ────────────────────────────────────────────────────────────
echo ""
echo "================================================="
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "All runs completed successfully."
else
    echo "${#FAILED_RUNS[@]} run(s) failed:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    echo "Check individual logs in $out/logs/ for details."
    if [[ -s "$FAILED_METHODS_FILE" ]]; then
        echo "Methods that failed for ALL datasets (candidates for removal from pipeline):"
        cat "$FAILED_METHODS_FILE"
    fi
fi
echo "================================================="