#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]];then echo "Usage: $0 SELECTED_LAMBDA";exit 2;fi
L=$1
ROOT=/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/FullVolume-NoPrompt
OUT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260905_CTV/FullVolume-NoPrompt/TrainResults/fivefold_cross_validation
PY=/home/wusi/miniconda3/envs/sam2/bin/python
export PYTHONPATH=/home/wusi/SAM2:${PYTHONPATH:-}
cd "$ROOT"
for F in 0 1 2 3 4;do
 D="$OUT/fold_${F}";mkdir -p "$D"
 ARGS=(train.py --fold "$F" --lambda-rank "$L" --epochs 100 --patience 10 --out "$D")
 [[ -f "$D/last.pth" ]] && ARGS+=(--resume)
 "$PY" "${ARGS[@]}" 2>&1 | tee -a "$D/train.log"
done
"$PY" select_best_fold.py --root "$OUT"