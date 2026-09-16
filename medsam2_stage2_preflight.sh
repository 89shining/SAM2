#!/usr/bin/env bash
set -euo pipefail

RESULT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage2-point/Preflight/fold_0
CODE=/home/wusi/MedSAM2/MyCodes/ESO/CTV/T-20260909/Stage2-point
CKPT=/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage1-mask/TrainResults/Mixed_K1_5/fold_0/checkpoints/best.pth

mkdir -p "$RESULT"
cd "$CODE"
export PYTHONPATH=/home/wusi/MedSAM2
export CUDA_VISIBLE_DEVICES=5
PY=/home/wusi/miniconda3/envs/medsam2/bin/python

"$PY" check_p0_equivalence.py --fold 0 --stage1-ckpt "$CKPT" \
  --output "$RESULT/p0_equivalence.json" > "$RESULT/preflight.log" 2>&1
"$PY" probe_stage2.py --fold 0 --patient-id 9 --stage1-ckpt "$CKPT" --budgets 0 1 \
  --output "$RESULT/micro_smoke.json" >> "$RESULT/preflight.log" 2>&1
