#!/bin/bash

########################################################
# Vast.ai launcher (DP)
#
# On our slurm cluster, run_cluster_*.sh created a docker container
# and ran the scripts in it. A Vast.ai instance is already a docker
# container (image: 25fallcsed490f/cluster:week4), so we run the
# scripts directly on the instance.
#
# Usage (at the repository root, inside tmux):
#   bash run_vastai_dp.sh
#
# Outputs
#   - logs/dp_{timestamp}.out : stdout / stderr of this script
#   - nsight_logs/dp_{timestamp}/ : Nsight logs (gpu_1, gpu_2)
#   - runs/dp_{timestamp}/ : checkpoints
########################################################

set -o pipefail

cd "$(dirname "$0")" || exit 1

export TIMESTAMP=$(date +"%Y%m%d_%H%M%S") # shared with scripts/run_*.sh
mkdir -p logs
LOG_FILE="logs/dp_${TIMESTAMP}.out"

{
    echo "[System] check instance environment..."
    bash scripts/check_env.sh || { echo "[System] environment check failed"; exit 1; }
    echo "[System] check instance environment finished"

    echo "[System] run train_cifar.py with 'run_dp.sh'..."
    bash scripts/run_dp.sh
    echo "[System] train ended"

    date
    echo "##### END #####"
} 2>&1 | tee "${LOG_FILE}"
