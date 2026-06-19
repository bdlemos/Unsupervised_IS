#!/bin/bash

# nohup ./run_pipeline.sh > pipeline_geral.log 2>&1 &
# Usage examples:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --methods "adaptive-perplexity,adaptive-v2-perplexity" --datasets "mpqa,sst2"
#   ./run_pipeline.sh --steps "1,2,3"
#   ./run_pipeline.sh --steps "selection,benchmark"

# Exit immediately if a command exits with a non-zero status
set -e

ROOT_DIR="/home/bernardo/projects"

# ── Default values ────────────────────────────────────────────────────────────
DEFAULT_METHODS="no-is,biois,perplexity-is,autoencoder-is,gmm-is,adaptive-perplexity,adaptive-v2-perplexity"

# Discover datasets automatically and sort them by the size of their texts.txt
DATASETS_DIR="$ROOT_DIR/unsupervised-is/resources/datasets"
if [[ -d "$DATASETS_DIR" ]]; then
    DEFAULT_DATASETS="$(python3 -c "
import os
d_dir = '$DATASETS_DIR'
items = []
for name in os.listdir(d_dir):
    p = os.path.join(d_dir, name)
    if os.path.isdir(p):
        t_path = os.path.join(p, 'texts.txt')
        sz = os.path.getsize(t_path) if os.path.isfile(t_path) else 0
        items.append((sz, name))
items.sort()
print(','.join([name for _, name in items]))
")"
else
    # Fallback: list sorted by texts.txt size at time of last update
    DEFAULT_DATASETS="mpqa,trec,sst2,twitter,vader_movie,movie_review,sst1,pang_movie,subj,yelp_reviews,wos5736,reuters90,webkb,wos11967,ohsumed,20ng,agnews,dblp,books,yelp_2013,medline"
fi

METHODS="$DEFAULT_METHODS"
DATASETS="$DEFAULT_DATASETS"
STEPS=""
INPUT_REP="tfidf"

# ── Step selector helper ──────────────────────────────────────────────────────
run_step() {
    local step_num="$1"
    local step_name="$2"
    if [[ -z "$STEPS" ]]; then
        return 0
    fi
    if [[ ",$STEPS," =~ ,$step_num, ]] || [[ ",$STEPS," =~ ,$step_name, ]]; then
        return 0
    fi
    return 1
}



# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --inputrep)
            INPUT_REP="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--methods <m1,m2,...>] [--datasets <d1,d2,...>] [--steps <s1,s2,...>]"
            exit 1
            ;;
    esac
done

echo "Methods  : $METHODS"
echo "Datasets : $DATASETS"
if [[ -n "$STEPS" ]]; then
    echo "Steps    : $STEPS"
else
    echo "Steps    : All (1, 2, 3, 4, 5)"
fi

# ── Step 1 ────────────────────────────────────────────────────────────────────
if run_step 1 selection; then
    echo "================================================="
    echo "Step 1: Running unsupervised selection"
    echo "================================================="
    cd "$ROOT_DIR/unsupervised-is"
    source venv/bin/activate
    bash bash/run_unsupervised_selection.sh --methods "$METHODS" --datasets "$DATASETS" --inputrep "$INPUT_REP"
    deactivate
fi

# ── Step 2 ────────────────────────────────────────────────────────────────────
if run_step 2 summary; then
    echo -e "\n================================================="
    echo "Step 2: Generating summary (read_selection_ci.py)"
    echo "================================================="
    cd "$ROOT_DIR/unsupervised-is"
    source venv/bin/activate
    python scripts/read_selection_ci.py
    deactivate
fi

# ── Step 3 ────────────────────────────────────────────────────────────────────
if run_step 3 benchmark; then
    echo -e "\n================================================="
    echo "Step 3: Running atcBench"
    echo "================================================="
    cd "$ROOT_DIR/atcBench"
    source venv/bin/activate
    bash run.sh --num-processes 1 --methods "$METHODS" --datasets "$DATASETS"
    deactivate
fi

# ── Step 4 ────────────────────────────────────────────────────────────────────
if run_step 4 metrics; then
    echo -e "\n================================================="
    echo "Step 4: Generating results CSV (Metrics)"
    echo "================================================="
    cd "$ROOT_DIR/atcBench"
    source venv/bin/activate
    mkdir -p resources/results
    python scripts/generate_results_csv.py -o resources/results/results_from_outputs.csv
    deactivate
fi

# ── Step 5 ────────────────────────────────────────────────────────────────────
if run_step 5 times; then
    echo -e "\n================================================="
    echo "Step 5: Generating results CSV (Times)"
    echo "================================================="
    cd "$ROOT_DIR/atcBench"
    source venv/bin/activate
    mkdir -p resources/results
    python scripts/generate_times_csv.py -o resources/results/times_from_outputs.csv
    deactivate
fi

echo -e "\n================================================="
echo "Pipeline execution completed successfully!"
echo "================================================="
