#!/usr/bin/env bash
set -euo pipefail

source /home/wusi/miniconda3/etc/profile.d/conda.sh
conda activate sam2
cd /home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260909/Stage2-point

export PYTHONPATH="/home/wusi/SAM2:/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260909/Stage2-point${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=6

OUT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage2-point/Preflight/fold_0/native_click_audit_5case_updated
mkdir -p "${OUT}"

/home/wusi/miniconda3/envs/sam2/bin/python pilot_d0_d5.py \
  --fold 0 \
  --stage1-ckpt /home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth \
  --checkpoint /home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth \
  --validation-plan /home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json \
  --patient-count 5 \
  --output-dir "${OUT}" \
  2>&1 | tee "${OUT}/audit.log"
