#!/bin/bash

# nohup ./run_pipeline.sh > pipeline_geral.log 2>&1 &
# Usage examples:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --methods "adaptive-perplexity,adaptive-v2-perplexity" --datasets "mpqa,sst2"

# Exit immediately if a command exits with a non-zero status
set -e

ROOT_DIR="/home/bernardo/projects"

# ── Default values ────────────────────────────────────────────────────────────
DEFAULT_METHODS="no-is,biois,perplexity-is,autoencoder-is,gmm-is,adaptive-perplexity,adaptive-v2-perplexity"

# Discover datasets automatically from the resources directory
DATASETS_DIR="$ROOT_DIR/unsupervised-is/resources/datasets"
if [[ -d "$DATASETS_DIR" ]]; then
    DEFAULT_DATASETS="$(ls -d "$DATASETS_DIR"/*/ 2>/dev/null | xargs -I{} basename {} | paste -sd ',')"
else
    # Fallback: list found at time of last update
    DEFAULT_DATASETS="20ng,agnews,books,dblp,medline,movie_review,mpqa,ohsumed,pang_movie,reuters90,sst1,sst2,subj,trec,vader_movie,webkb,wos11967,wos5736,yelp_2013,yelp_reviews"
fi

METHODS="$DEFAULT_METHODS"
DATASETS="$DEFAULT_DATASETS"


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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--methods <m1,m2,...>] [--datasets <d1,d2,...>]"
            exit 1
            ;;
    esac
done

echo "Methods  : $METHODS"
echo "Datasets : $DATASETS"

echo "================================================="
echo "Step 1: Running unsupervised selection"
echo "================================================="
cd "$ROOT_DIR/unsupervised-is"
source venv/bin/activate
bash bash/run_unsupervised_selection.sh --methods "$METHODS" --datasets "$DATASETS"
deactivate

echo -e "\n================================================="
echo "Step 2: Generating summary (read_selection_ci.py)"
echo "================================================="
cd "$ROOT_DIR/unsupervised-is"
source venv/bin/activate
python scripts/read_selection_ci.py
deactivate

echo -e "\n================================================="
echo "Step 3: Running atcBench"
echo "================================================="
cd "$ROOT_DIR/atcBench"
source venv/bin/activate
bash run.sh --num-processes 5 --methods "$METHODS" --datasets "$DATASETS"
deactivate


echo -e "\n================================================="
echo "Step 4: Generating results CSV (Metrics)"
echo "================================================="
cd "$ROOT_DIR/atcBench"
source venv/bin/activate
python scripts/generate_results_csv.py

echo -e "\n================================================="
echo "Step 5: Generating results CSV (Times)"
echo "================================================="
python scripts/generate_times_csv.py
deactivate

echo -e "\n================================================="
echo "Pipeline execution completed successfully!"
echo "================================================="
