#!/usr/bin/env bash
set -euo pipefail

RESULT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage2-point/Preflight/fold_0
CODE=/home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point
CKPT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth

mkdir -p "$RESULT"
cd "$CODE"
PYTHONPATH=/home/wusi/MedSAM2 CUDA_VISIBLE_DEVICES=5 /home/wusi/miniconda3/envs/medsam2/bin/python \
  probe_stage2.py --fold 0 --patient-id 9 --stage1-ckpt "$CKPT" --budgets 0 1 3 5 \
  --output "$RESULT/micro_smoke_T0_1_3_5.json" > "$RESULT/probe_T0_1_3_5.log" 2>&1
