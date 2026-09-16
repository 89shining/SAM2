#!/usr/bin/env bash
set -euo pipefail

source /home/wusi/miniconda3/etc/profile.d/conda.sh
conda activate nninteractive
cd /home/wusi/nnInteractive/MyCodes/ESO/CTV/T-20260909/Stage2-point

export PYTHONPATH="/home/wusi/nnInteractive:/home/wusi/nnInteractive/MyCodes/ESO/CTV/T-20260909/Stage2-point${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=5

RUN_ROOT=/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults
RUN_DIR="${RUN_ROOT}/Workflow_K3_T0_5/fold_0"
mkdir -p "${RUN_DIR}"

exec 9>/tmp/nninteractive_stage2_formal_fold0.lock
flock -n 9 || { echo "nnInteractive Stage2 fold0 already running"; exit 1; }
exec > >(tee -a "${RUN_DIR}/launch.log") 2>&1

exec /home/wusi/miniconda3/envs/nninteractive/bin/python train.py \
  --fold 0 \
  --epochs 100
