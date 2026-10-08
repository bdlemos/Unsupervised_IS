#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Uso: bash info_scripts/prepare_dataset_jina.sh <dataset>" >&2
    exit 2
fi

dataset_name="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
data_dir="${DATASETS_DIR:-$repo_root/datasets}"
python_bin="${PYTHON:-python}"

"$python_bin" "$script_dir/download_datasets.py" "$dataset_name" --data-dir "$data_dir"
DATASETS_DIR="$data_dir" "$python_bin" "$script_dir/generate_jina_v5.py" \
    --datasets "$dataset_name" --folds 10
