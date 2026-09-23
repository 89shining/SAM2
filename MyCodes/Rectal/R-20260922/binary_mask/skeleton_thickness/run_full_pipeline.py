#!/usr/bin/env python3
"""One-shot formal R-20260922 pipeline; training then validation lock then test.

This intentionally never starts an inference stage before all checkpoints have
finished.  Test metrics are written only after validation-created lock files
exist and are never used to modify those locks.
"""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path

CODE=Path(__file__).parent
DATA=Path('/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask')
PY=Path('/home/wusi/miniconda3/envs/sam2/bin/python')
p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--gpus',type=int,nargs=2,required=True); a=p.parse_args()
def run(args):
    print('RUN', ' '.join(map(str,args)), flush=True); subprocess.run(list(map(str,args)),check=True)

# Two A100s are deliberately reserved for this experiment only.
reserved=[str(x) for x in range(8) if x not in a.gpus]
run([PY,CODE/'schedule_training.py','--run-root',a.run_root,'--reserve-gpus',*reserved,'--max-parallel','2'])
run([PY,CODE/'run_validation_inference.py','--run-root',a.run_root,'--gpus',*a.gpus])
for kind in ('pos','neg'):
    run([PY,CODE/'select_branch_thickness_validation.py','--kind',kind,'--data-root',DATA,
         '--split-json',a.run_root/'checkpoints'/kind/'t00mm'/'split.json',
         '--cache-root',a.run_root/'probability_cache_validation','--metrics-root',a.run_root/'metrics'])
run([PY,CODE/'plot_validation_profiles.py','--metrics-root',a.run_root/'metrics'])
if not all((a.run_root/'metrics'/f'LOCKED_{k.upper()}_VALIDATION_CONFIG.json').is_file() for k in ('pos','neg')):
    raise RuntimeError('Validation locks were not created; refusing test.')
run([PY,CODE/'run_test_inference.py','--run-root',a.run_root,'--gpus',*a.gpus])
fusion=CODE.parent/'fusion_test'/'run_fusion_test.py'
run([PY,fusion,'--data-root',DATA,'--cache-root',a.run_root/'probability_cache_test',
     '--selection-root',a.run_root/'metrics','--output-root',a.run_root.parent/'fusion_test'])
(a.run_root/'PIPELINE_COMPLETE').touch()
