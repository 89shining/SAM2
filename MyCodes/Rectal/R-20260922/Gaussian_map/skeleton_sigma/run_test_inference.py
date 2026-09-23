#!/usr/bin/env python3
"""Run test caches for every validation-locked per-sigma configuration."""
from __future__ import annotations
import argparse, subprocess, json
from pathlib import Path
T=(1,2,3,4); F=(0,.25,.5,.75,1.)
p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); p.add_argument('--code-root',type=Path,default=Path(__file__).parent); p.add_argument('--data-root',type=Path,default=Path('/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask')); p.add_argument('--python',type=Path,default=Path('/home/wusi/miniconda3/envs/sam2/bin/python')); p.add_argument('--gpus',type=int,nargs='+',default=[0]); a=p.parse_args()
i=0
locks={k:json.loads((a.run_root/'metrics'/f'LOCKED_{k.upper()}_VALIDATION_CONFIG.json').read_text())['locked'] for k in ('pos','neg')}
for f in F:
  tag=f'p{int(round(f*100)):03d}'
  # Test only final validation-locked POS and NEG configurations.
  for k,t in ((k,int(float(locks[k]['sigma_mm']))) for k in ('pos','neg')):
    ck=a.run_root/'checkpoints'/k/f's{t:02d}mm'/'checkpoints'/'best.pth'; split=a.run_root/'checkpoints'/k/f's{t:02d}mm'/'split.json'
    cmd=['env',f'CUDA_VISIBLE_DEVICES={a.gpus[i%len(a.gpus)]}',str(a.python),str(a.code_root/'infer_gaussian_sigma.py'),'--kind',k,'--data-root',str(a.data_root),'--split-json',str(split),'--subset','test','--checkpoint',str(ck),'--output-root',str(a.run_root/'probability_cache_test'/f'fraction_{tag}'/k),'--sigmas-mm',str(t),'--input-size','512','--prompt-fraction',str(f),'--segment-fraction','0.5','--device','cuda']; subprocess.run(cmd,check=True); i+=1
