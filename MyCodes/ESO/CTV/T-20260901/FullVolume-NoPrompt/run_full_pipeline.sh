#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wusi/SAM2/MyTrain/MyCodes/ESO/CTV/T-20260901/FullVolume-NoPrompt
RESULT=/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260905_CTV/FullVolume-NoPrompt
PY=/home/wusi/miniconda3/envs/sam2/bin/python
export PYTHONPATH=/home/wusi/SAM2:${PYTHONPATH:-}
cd "$ROOT"
bash run_lambda_search.sh
L=$("$PY" -c "import json;print(json.load(open('$RESULT/TrainResults/lambda_hyperparameter_search/lambda_selection.json'))['selected']['lambda_rank'])")
bash run_fivefold.sh "$L"
CKPT=$("$PY" -c "import json;print(json.load(open('$RESULT/TrainResults/fivefold_cross_validation/selected_best_fold.json'))['checkpoint'])")
"$PY" test.py --checkpoint "$CKPT" --out "$RESULT/TestResults"
"$PY" evaluate_test.py --pred-dir "$RESULT/TestResults"
"$PY" -c "import json,pathlib; pathlib.Path('$RESULT/pipeline_complete.json').write_text(json.dumps({'selected_lambda':float('$L'),'selected_checkpoint':'$CKPT'},indent=2))"