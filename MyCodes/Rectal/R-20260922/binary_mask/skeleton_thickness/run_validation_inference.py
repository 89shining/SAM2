#!/usr/bin/env python3
"""Run fixed-best-checkpoint validation inference for all trained thicknesses."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
T=(0,2,4,6,8,10,12)
p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--code-root',type=Path,default=Path(__file__).parent); p.add_argument('--data-root',type=Path,default=Path('/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask')); p.add_argument('--python',type=Path,default=Path('/home/wusi/miniconda3/envs/sam2/bin/python')); p.add_argument('--gpus',type=int,nargs='+',default=[0]); a=p.parse_args()
for i,(k,t) in enumerate((k,t) for k in ('pos','neg') for t in T):
    ck=a.run_root/'checkpoints'/k/f't{t:02d}mm'/'checkpoints'/'best.pth'; split=a.run_root/'checkpoints'/k/f't{t:02d}mm'/'split.json'
    if not ck.is_file(): raise FileNotFoundError(ck)
    cmd=['env',f'CUDA_VISIBLE_DEVICES={a.gpus[i%len(a.gpus)]}',str(a.python),str(a.code_root/'infer_skeleton_thickness.py'),'--kind',k,'--data-root',str(a.data_root),'--split-json',str(split),'--subset','validation','--checkpoint',str(ck),'--output-root',str(a.run_root/'probability_cache_validation'/k),'--thicknesses-mm',str(t),'--input-size','512','--prompt-fraction','0.5','--segment-fraction','0.5','--device','cuda']
    subprocess.run(cmd,check=True)
