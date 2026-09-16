#!/usr/bin/env bash
set -euo pipefail

CODE=/home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point
ROOT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV
CKPT=$ROOT/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth
OUT=$ROOT/Stage2-point/Preflight/fold_0/native_click_audit_5case

mkdir -p "$OUT"
cd "$CODE"
PYTHONPATH=/home/wusi/MedSAM2 CUDA_VISIBLE_DEVICES=5 /home/wusi/miniconda3/envs/medsam2/bin/python \
  pilot_d0_d5.py --fold 0 --stage1-ckpt "$CKPT" --checkpoint "$CKPT" \
  --validation-plan /home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults/validation_prompt_plan.json \
  --patient-count 5 --output-dir "$OUT" > "$OUT/audit.log" 2>&1
