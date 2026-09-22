#!/bin/bash

########################################################
# Pre-flight check for the Vast.ai instance.
# It is called by run_vastai_*.sh before training.
#
# On our slurm cluster, the hardware & dataset were fixed by TAs.
# On Vast.ai, every instance can be different, so we check
#   1. the number of GPUs
#   2. PyTorch can run CUDA kernels on every GPU, and GPU-to-GPU copies are correct
#   3. Nsight Systems CLI (nsys)
#   4. dataset
#   5. shared memory size (used by NCCL)
########################################################

REQUIRED_GPUS=${REQUIRED_GPUS:-2}
DATASET_DIR="${DATASET_DIR:-dataset}"
STATUS=0

echo "[Check] hostname : $(hostname)"
echo "[Check] date     : $(date)"

## 1. GPU
echo "[Check] GPU list"
nvidia-smi -L
NUM_GPUS_FOUND=$(nvidia-smi -L | grep -c "^GPU")
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo "[Check] driver version: ${DRIVER_VERSION}"
if [ "${NUM_GPUS_FOUND}" -lt "${REQUIRED_GPUS}" ]; then
    echo "[Error] This lab needs at least ${REQUIRED_GPUS} GPUs, but ${NUM_GPUS_FOUND} GPU(s) found."
    echo "[Error] Please rent a Vast.ai instance with ${REQUIRED_GPUS} or more GPUs."
    STATUS=1
fi

## 2. PyTorch + CUDA
python - <<'EOF' || STATUS=1
import sys
import torch
print(f"[Check] torch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("[Error] torch.cuda.is_available() is False")
    sys.exit(1)
failed = False
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    try:
        x = torch.randn(64, 64, device=f"cuda:{i}")
        (x @ x).sum().item()
        print(f"[Check] cuda:{i} {name} (sm_{cap[0]}{cap[1]}) OK")
    except Exception as e:
        failed = True
        print(f"[Error] cuda:{i} {name} (sm_{cap[0]}{cap[1]}) can not run PyTorch kernels: {e}")
if failed:
    print("[Error] This PyTorch build does not support the GPU. (e.g. RTX 50xx / Blackwell is not supported)")
    print("[Error] Please rent another instance (RTX 3090 / RTX 4090 / A5000 / A6000 ...).")
    sys.exit(1)
EOF

## 2-1. GPU peer-to-peer (P2P) copy
# Some multi-GPU hosts report P2P as available, but GPU-to-GPU copies return wrong values.
# DP (torch.nn.DataParallel) copies tensors between GPUs directly, so its loss becomes NaN on such hosts.
# (DDP uses NCCL, which is usually not affected.)
if [ "${NUM_GPUS_FOUND}" -ge 2 ]; then
    python - <<'EOF'
import sys
import torch
if torch.cuda.device_count() < 2:
    sys.exit(0)
x = torch.randn(1 << 20, device="cuda:0")
ok_01 = torch.equal(x.to("cuda:1").cpu(), x.cpu())
y = torch.randn(1 << 20, device="cuda:1")
ok_10 = torch.equal(y.to("cuda:0").cpu(), y.cpu())
p2p = torch.cuda.can_device_access_peer(0, 1)
if ok_01 and ok_10:
    print(f"[Check] GPU0 <-> GPU1 copy OK (P2P access: {p2p})")
    sys.exit(0)
print(f"[Error] GPU0 <-> GPU1 copy returns WRONG values (P2P access: {p2p}).")
print("[Error] This host has broken GPU peer-to-peer. DP with 2 GPUs will produce NaN loss.")
print("[Error] Please destroy this instance and rent another one.")
print("[Error] (To run anyway, set ALLOW_BROKEN_P2P=1)")
sys.exit(1)
EOF
    if [ $? -ne 0 ] && [ "${ALLOW_BROKEN_P2P:-0}" != "1" ]; then
        STATUS=1
    fi
fi

## 3. nsys
if command -v nsys > /dev/null 2>&1; then
    echo "[Check] $(nsys --version)"
else
    echo "[Error] nsys is not found. Run 'bash scripts/init.sh' first."
    STATUS=1
fi

## 4. dataset
for d in "${DATASET_DIR}/cifar10" "${DATASET_DIR}/cifar10_images"; do
    if [ -d "$d" ]; then
        echo "[Check] dataset: $d"
    else
        echo "[Error] dataset '$d' is not found. Run 'bash scripts/init.sh' first."
        STATUS=1
    fi
done

## 5. shared memory
echo "[Check] shared memory: $(df -h /dev/shm | tail -n 1 | awk '{print $2}')"

exit ${STATUS}
