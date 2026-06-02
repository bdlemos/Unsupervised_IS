#!/bin/bash

# nohup ./run_pipeline.sh > pipeline_geral.log 2>&1 &


# Exit immediately if a command exits with a non-zero status
set -e

ROOT_DIR="/home/bernardo/projects"

echo "================================================="
echo "Step 1: Running unsupervised selection"
echo "================================================="
cd "$ROOT_DIR/unsupervised-is"
source venv/bin/activate
bash bash/run_unsupervised_selection.sh
deactivate

echo -e "\n================================================="
echo "Step 2: Generating summary (read_selection_ci.py)"
echo "================================================="
cd "$ROOT_DIR/unsupervised-is"
source venv/bin/activate
python read_selection_ci.py
deactivate

echo -e "\n================================================="
echo "Step 3: Running atcBench"
echo "================================================="
cd "$ROOT_DIR/atcBench"
source env/bin/activate
bash run.sh 1
deactivate

echo -e "\n================================================="
echo "Step 4: Generating results CSV (Metrics)"
echo "================================================="
cd "$ROOT_DIR/atcBench"
source env/bin/activate
python scripts/generate_results_csv.py

echo -e "\n================================================="
echo "Step 5: Generating results CSV (Times)"
echo "================================================="
python scripts/generate_times_csv.py
deactivate

echo -e "\n================================================="
echo "Pipeline execution completed successfully!"
echo "================================================="
