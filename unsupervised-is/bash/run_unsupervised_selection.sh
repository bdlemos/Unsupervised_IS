#!/usr/bin/env bash
# run command -> nohup ./bash/run_unsupervised_selection.sh > run.log 2>&1 &
# bash bash/run_unsupervised_selection.sh 

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

source "$WORKDIR/venv/bin/activate"

datain="$WORKDIR/resources/datasets"
out="$WORKDIR/resources/outsel"

mkdir -p "$out"

# datasets=(mpqa reuters90 sst1 ohsumed twitter webkb)
# datasets=(yelp_reviews sst2 dblp books acm)

datasets=(mpqa reuters90 sst1 ohsumed twitter webkb yelp_reviews sst2 dblp acm )
# methods=(perplexity-is gmm-is no-is random-is autoencoder-is biois)
methods=(adaptive-v2-perplexity adaptive-perplexity)

for dataset in "${datasets[@]}"; do
    echo "$dataset"
    for method in "${methods[@]}"; do
        echo "$method"
        python scripts/run_generateSplit.py -d "$dataset" -m "$method" --datain "$datain" --out "$out"
    done
done
