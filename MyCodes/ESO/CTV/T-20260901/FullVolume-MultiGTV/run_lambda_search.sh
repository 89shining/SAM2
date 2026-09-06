#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/FullVolume-MultiGTV
OUT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260905_CTV/FullVolume-MultiGTV/TrainResults/lambda_hyperparameter_search
PY=/home/wusi/miniconda3/envs/sam2/bin/python
export PYTHONPATH=/home/wusi/SAM2:${PYTHONPATH:-}
cd "$ROOT"
for L in 0.0 0.01 0.05 0.10 0.20; do
 D="$OUT/lambda_${L}";mkdir -p "$D"
 ARGS=(train.py --fold 0 --lambda-rank "$L" --epochs 100 --patience 10 --out "$D")
 [[ -f "$D/last.pth" ]] && ARGS+=(--resume)
 "$PY" "${ARGS[@]}" 2>&1 | tee -a "$D/train.log"
done
"$PY" select_best_lambda.py --root "$OUT"