#!/usr/bin/env bash
set -euo pipefail

source /home/wusi/miniconda3/etc/profile.d/conda.sh
conda activate medsam2
cd /home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point

export PYTHONPATH="/home/wusi/MedSAM2:/home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=5

RUN_ROOT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults_native_smoke
LOG_DIR="${RUN_ROOT}/Stage2_T0_5/fold_0"
mkdir -p "${LOG_DIR}"

# Keep exactly one process for this disposable smoke run.  `exec` makes the
# tmux session own the Python process directly, so terminating the session
# cannot leave a second trainer writing the same checkpoint directory.
exec 9>/tmp/medsam2_stage2_native_epoch1_smoke.lock
flock -n 9 || { echo "smoke already running"; exit 1; }
exec > >(tee "${LOG_DIR}/launch.log") 2>&1

exec /home/wusi/miniconda3/envs/medsam2/bin/python train.py \
  --fold 0 \
  --stage1-ckpt /home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth \
  --epochs 1 \
  --output-root "${RUN_ROOT}" \
  --no-resume
