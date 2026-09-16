#!/usr/bin/env bash
set -euo pipefail

CODE=/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260909/Stage2-point
ROOT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV
CKPT=$ROOT/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth
OUT=$ROOT/Stage2-point/Preflight/fold_0/d0_d5_pilot_stage1

mkdir -p "$OUT"
cd "$CODE"
PYTHONPATH=/home/wusi/SAM2 CUDA_VISIBLE_DEVICES=4 /home/wusi/miniconda3/envs/sam2/bin/python \
  pilot_d0_d5.py --fold 0 --stage1-ckpt "$CKPT" --checkpoint "$CKPT" \
  --validation-plan "$ROOT/Stage1-mask/TrainResults/validation_prompt_plan.json" \
  --patient-count 5 --output-dir "$OUT" > "$OUT/pilot.log" 2>&1
