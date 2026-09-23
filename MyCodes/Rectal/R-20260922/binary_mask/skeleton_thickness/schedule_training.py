#!/usr/bin/env python3
"""GPU scheduler for 14 independent skeleton-thickness models.

It never starts unless preflight passed, reuses the exact 0916 99/11 split,
and passes --skip-test: official inference starts only after validation locks m.
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time
from collections import deque
from pathlib import Path

THICKNESSES=(0,2,4,6,8,10,12); BRANCHES=('pos','neg')
SPLIT=Path('/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260916/radius_propagation_distance/small.pth/checkpoints/neg/r20mm/split.json')

def free_gpus(reserved: set[int], minimum: int):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'], text=True)
    ans=[]
    for row in text.splitlines():
        i, mem, util=[int(x.strip()) for x in row.split(',')]
        if i not in reserved and mem>=minimum and util<=20: ans.append(i)
    return ans

p=argparse.ArgumentParser()
p.add_argument('--run-root',type=Path,required=True); p.add_argument('--code-root',type=Path,default=Path(__file__).parent)
p.add_argument('--data-root',type=Path,default=Path('/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask'))
p.add_argument('--python',type=Path,default=Path('/home/wusi/miniconda3/envs/sam2/bin/python'))
p.add_argument('--reserve-gpus',type=int,nargs='*',default=[]); p.add_argument('--max-parallel',type=int,default=4); p.add_argument('--min-free-mib',type=int,default=16000)
p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
marker=a.run_root/'PREFLIGHT_PASS'
if not marker.is_file(): raise SystemExit(f'preflight required: {marker}')
if not SPLIT.is_file(): raise SystemExit(f'missing reference split: {SPLIT}')
tasks=deque((t,b) for t in THICKNESSES for b in BRANCHES); active={}; retries={}
while tasks or active:
    for gpu in free_gpus(set(a.reserve_gpus)|set(active), a.min_free_mib):
        if len(active)>=a.max_parallel or not tasks: break
        t,b=tasks.popleft(); out=a.run_root/'checkpoints'/b/f't{t:02d}mm'; best=out/'checkpoints'/'best.pth'
        if best.is_file(): continue
        out.mkdir(parents=True,exist_ok=True); shutil.copy2(SPLIT,out/'split.json')
        cmd=[str(a.python),str(a.code_root/f'fullmask_{b}.py'),'--data-root',str(a.data_root),'--output-root',str(out),'--test-results-dir',str(a.run_root/'unused_train_test'/b/f't{t:02d}mm'),'--init-checkpoint','/home/wusi/SAM2/checkpoints/sam2.1_hiera_small.pt','--input-size','512','--skeleton-thickness-mm',str(t),'--max-epochs','60','--patience','10','--warmup-epochs','5','--early-stop-start-epoch','5','--train-prompt-keep-min','0.30','--train-prompt-keep-max','0.70','--eval-prompt-keep','0.50','--train-temporal-window','9','--skip-test','--num-workers','4','--device','cuda']
        print('GPU',gpu,'TASK',t,b,' '.join(cmd),flush=True)
        if a.dry_run: continue
        env=dict(__import__('os').environ, CUDA_VISIBLE_DEVICES=str(gpu)); log=(a.run_root/'logs'/f'{b}_t{t:02d}.log'); log.parent.mkdir(parents=True,exist_ok=True)
        active[gpu]=(t,b,subprocess.Popen(cmd,stdout=log.open('a'),stderr=subprocess.STDOUT,env=env))
    for gpu,item in list(active.items()):
        t,b,proc=item; status=proc.poll()
        if status is None: continue
        del active[gpu]
        if status!=0:
            retries[(t,b)]=retries.get((t,b),0)+1
            if retries[(t,b)]<=3: tasks.append((t,b))
            else: raise SystemExit(f'permanent failure {t}/{b}; see logs')
    time.sleep(20)
(a.run_root/'TRAINING_COMPLETE').touch()
