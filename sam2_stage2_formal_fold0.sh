#!/usr/bin/env bash
set -euo pipefail
source /home/wusi/miniconda3/etc/profile.d/conda.sh
conda activate sam2
cd /home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260909/Stage2-point
export PYTHONPATH="/home/wusi/SAM2:/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260909/Stage2-point${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=4
RUN_ROOT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage2-point/TrainResults
RUN_DIR="${RUN_ROOT}/Stage2_T0_5/fold_0"
mkdir -p "${RUN_DIR}"
exec 9>/tmp/sam2_stage2_formal_fold0.lock
flock -n 9 || { echo "SAM2 formal fold0 already running"; exit 1; }
exec > >(tee -a "${RUN_DIR}/launch.log") 2>&1
exec /home/wusi/miniconda3/envs/sam2/bin/python train.py \
  --fold 0 \
  --stage1-ckpt /home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth \
  --epochs 100 \
  --resume
