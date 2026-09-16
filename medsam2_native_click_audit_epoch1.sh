#!/usr/bin/env bash
set -euo pipefail

source /home/wusi/miniconda3/etc/profile.d/conda.sh
conda activate medsam2
cd /home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point

export PYTHONPATH="/home/wusi/MedSAM2:/home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=5

OUT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults_native_smoke/Stage2_T0_5/fold_0/epoch1_click_audit_5case
mkdir -p "${OUT}"

exec 9>/tmp/medsam2_epoch1_click_audit.lock
flock -n 9 || { echo "epoch-1 click audit already running"; exit 1; }
exec > >(tee "${OUT}/audit.log") 2>&1

exec /home/wusi/miniconda3/envs/medsam2/bin/python pilot_d0_d5.py \
  --fold 0 \
  --stage1-ckpt /home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth \
  --checkpoint /home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults_native_smoke/Stage2_T0_5/fold_0/checkpoints/best.pth \
  --validation-plan /home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json \
  --patient-count 5 \
  --output-dir "${OUT}"
