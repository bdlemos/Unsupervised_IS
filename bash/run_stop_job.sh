#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Your workload
bash "${SCRIPT_DIR}/run_aggressive_rates.sh"

# Stop this pod when done
# runpodctl pod stop "$RUNPOD_POD_ID"
runpodctl pod stop vk118sjvcmdu3z
