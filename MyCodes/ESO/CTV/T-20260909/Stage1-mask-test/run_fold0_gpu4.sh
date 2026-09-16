#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260909/Stage1-mask-test
OUT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TestResults/main_deterministic/fold_0_only
PLAN=/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/prompt_plans/stage1_prompt_plan.json
mkdir -p "$OUT"
python3 - <<'PY' > "$OUT/run_manifest.json"
import json
print(json.dumps({"model":"SAM2","scope":"fold_0_only","plan":"/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/prompt_plans/stage1_prompt_plan.json","physical_gpu":4,"command":"run_stage1_evaluation.py --fold 0 --device cuda:0 --save-predictions"}, indent=2))
PY
cd "$ROOT"
CUDA_VISIBLE_DEVICES=4 /home/wusi/miniconda3/envs/sam2/bin/python run_stage1_evaluation.py --fold 0 --allow-legacy-fold0-checkpoint --device cuda:0 --save-predictions 2>&1 | tee "$OUT/run.log"
