#!/bin/bash

########################################################
# Initialize the Vast.ai instance. Run it ONCE per instance at the repository root.
#   bash scripts/init.sh
########################################################

cd "$(dirname "$0")/.." || exit 1

mkdir -m 777 -p runs
mkdir -m 777 -p logs
mkdir -m 777 -p nsight_logs

########################################################
# Nsight Systems CLI (nsys)
# Our docker image (25fallcsed490f/cluster:week4) already has nsys.
# If you use another image (e.g. Vast.ai PyTorch template), we install it
# from NVIDIA devtools apt repository, same as the Dockerfile.
########################################################

if ! command -v nsys > /dev/null 2>&1; then
    echo "[init] nsys not found. Install nsight-systems-cli..."
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    set -e
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y --no-install-recommends ca-certificates curl gnupg
    arch="$(dpkg --print-architecture)"
    ver="$(. /etc/os-release; echo ${VERSION_ID} | tr -d .)"
    curl -fsSL "https://developer.download.nvidia.com/devtools/repos/ubuntu${ver}/${arch}/nvidia.pub" \
        | ${SUDO} gpg --dearmor --yes -o /usr/share/keyrings/nvidia-devtools.gpg
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu${ver}/${arch}/ /" \
        | ${SUDO} tee /etc/apt/sources.list.d/nvidia-devtools.list > /dev/null
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y --no-install-recommends nsight-systems-cli
    set +e
fi
nsys --version

pip install -r requirements.txt

########################################################
# Unlike our slurm cluster, a Vast.ai instance does not have a shared dataset directory.
# So we download CIFAR-10 to the instance's local disk (./dataset, ignored by git).
#   dataset/cifar10        : torchvision format (for DP / DDP)
#   dataset/cifar10_images : png image folders   (for DALI)
# Saving 60,000 png images takes a few minutes. It is skipped if the dataset already exists.
#
# The original download server (www.cs.toronto.edu) is sometimes very slow,
# so we download the same file from our GitHub Release (md5 is checked).
# If it fails, the original server is used.
########################################################

DATASET_DIR="${DATASET_DIR:-dataset}"
REPO_URL="https://github.com/csed490f2026/CSED490F_DDP_DALI_Training"
export CIFAR10_URL="${CIFAR10_URL:-${REPO_URL}/releases/download/cifar10/cifar-10-python.tar.gz}"

python my_lib/init_dataset.py --seed 42 --dataset_dir "$DATASET_DIR"
