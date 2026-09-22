# CSED490F Lab 4: DDP / DALI Implementation & Profiling (Vast.ai)

This lab used to run on our Slurm cluster (`sbatch run_cluster_*.sh`). From this semester it runs on a **Vast.ai GPU instance**.
The things you implement are the same as before (Problem 0 ~ 7). Only the way you run the code has changed.

| | Slurm cluster (before) | Vast.ai (now) |
|---|---|---|
| Environment | `run_cluster_*.sh` ran `docker run` | The instance itself is a container running our image |
| Dataset | pre-downloaded in `/home/dataset`, mounted at `/DATA` | `bash scripts/init.sh` downloads it to `./dataset` |
| Run | `sbatch run_cluster_dp.sh` | `bash run_vastai_dp.sh` (inside tmux) |
| Logs | `slurm_logs/{job_id}.out` | `logs/{mode}_{timestamp}.out` |
| Download results | `scp user@141.223.181.103:...` | `scp -P <PORT> root@<IP>:...` |
| When finished | job ends automatically | **Destroy the instance yourself** (you pay until you do) |

## 1. Rent an instance

1. Register your SSH public key on the Vast.ai console (**Account → Keys**).
2. Search for an instance using these filters:
   - **GPUs: 2**. RTX 3090 is recommended because the given `.nsys-rep` files were recorded on RTX 3090s. RTX 4090, A5000 and A6000 also work.
   - **Do NOT rent RTX 50xx / B200 (Blackwell) GPUs.** Our image uses PyTorch 2.4.1 + CUDA 11.8, which cannot run on them.
   - **Max CUDA version ≥ 12.1**, which DALI (`nvidia-dali-cuda120`) requires.
   - **Disk ≥ 50 GB**. The image is about 20 GB after it is unpacked.
3. Template / image settings:
   - Image: `25fallcsed490f/cluster:week4`
   - Launch mode: **SSH**
4. The first boot pulls the image (about 8 GB), which can take several minutes.

## 2. Setup (once per instance)

Use the SSH command shown by the **Connect** button on your instance card:

```bash
ssh -p <PORT> root@<IP>

mkdir -p /workspace && cd /workspace
git clone https://github.com/csed490f2026/CSED490F_DDP_DALI_Training.git
cd CSED490F_DDP_DALI_Training

bash scripts/init.sh    # pip packages (+ nsys if missing) and the CIFAR-10 download
```

`init.sh` downloads CIFAR-10 from [our GitHub Release](https://github.com/csed490f2026/CSED490F_DDP_DALI_Training/releases/tag/cifar10), which takes about 10 seconds. If that fails, it downloads from the original server (www.cs.toronto.edu) instead, which is often very slow (around 50 KB/s, almost an hour for 170 MB). The file is the same either way: its md5 is checked.

`scripts/init.sh` creates:

- `dataset/cifar10`: torchvision format, used by DP and DDP
- `dataset/cifar10_images`: PNG folders, used by DALI

Saving the PNG images takes a few minutes. If `init.sh` is interrupted, run it again and it will prepare the dataset from scratch.

## 3. Run

Training runs for a while. Run it inside `tmux` so that it survives an SSH disconnection. Vast.ai SSH sessions usually open inside tmux already; if yours doesn't, run `tmux new -s lab4`.

```bash
bash run_vastai_dp.sh         # Problem 0 : DP (given) + your Nsight command
bash run_vastai_ddp.sh        # Problem 1~5 : DDP
bash run_vastai_ddp_dali.sh   # Problem 6~7 : DDP + DALI
```

Each script first checks the instance (`scripts/check_env.sh`: GPUs, PyTorch/CUDA, GPU-to-GPU copy, nsys, dataset), then trains with 1 GPU and then with 2 GPUs.

| Output | Path |
|---|---|
| stdout / stderr | `logs/{mode}_{timestamp}.out` |
| Nsight logs | `nsight_logs/{mode}_{timestamp}/gpu_1.nsys-rep`, `gpu_2.nsys-rep` |
| checkpoints | `runs/{mode}_{timestamp}/gpu_{1,2}/` |

## 4. Download the results & destroy the instance

On **your local device**:

```bash
scp -P <PORT> -r root@<IP>:/workspace/CSED490F_DDP_DALI_Training/nsight_logs ./nsight_logs
```

Open the `.nsys-rep` files in Nsight Systems on your local device. The local Nsight Systems version must be the same as or newer than `nsys --version` on the instance.

**When you are done, destroy the instance on the Vast.ai console.** A *stopped* instance is still charged for its disk. Download your results before you destroy it, because all files on the instance are deleted.

## 5. Implementation (same as before)

| Problem | File | |
|---|---|---|
| 0 | `scripts/launch.sh` | Nsight log generation command |
| 1, 2, 3 | `handler/DDP/utils.py` | `run_process`, `initialize_group`, `destroy_process` |
| 4 | `handler/DDP/model.py` | `model_to_DDP` |
| 5 | `handler/DDP/cifar10_loader.py` | `get_DDP_loader` |
| 6, 7 | `handler/DALI/cifar10_loader.py` | `CifarPipeline`, `get_DALI_loader` |

## Troubleshooting

- **`[Error] ... can not run PyTorch kernels` / `no kernel image is available`**: the GPU is not supported by our image (e.g. RTX 50xx). Rent another instance.
- **`[Error] GPU0 <-> GPU1 copy returns WRONG values`**: some multi-GPU hosts have broken GPU peer-to-peer. There, DP with 2 GPUs trains on corrupted data and its loss becomes NaN. Destroy the instance and rent another one.
- **DDP hangs at the start or at the first iteration**: some hosts have broken GPU peer-to-peer. Run `export NCCL_P2P_DISABLE=1`, then run the script again.
- **NCCL error about shared memory (`/dev/shm`)**: run `export NCCL_SHM_DISABLE=1`, then run the script again. `check_env.sh` prints the shared memory size.
- **nsys warns about CPU sampling or `perf_event_paranoid`**: Vast.ai instances are unprivileged containers, so CPU sampling is not allowed. You can ignore this warning; the CUDA / NVTX traces are still recorded.
