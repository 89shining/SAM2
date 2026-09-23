#!/usr/bin/env python3
"""Validation-only selection: fixed best epoch -> m*(sigma) -> sigma*."""
from __future__ import annotations
import argparse,csv,json,math,re
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
SIGMAS=(1,2,3,4); GATES=(0,5,10,15,20,30,40,60,math.inf); EPS=1e-12
def num(x): return int(re.search(r'(\d+)$',x.name).group(1))
def read(p):
 i=sitk.ReadImage(str(p)); return sitk.GetArrayFromImage(i).astype(bool),i
def dice(a,b):
 d=int(a.sum()+b.sum()); return 1. if d==0 else float(2*(a&b).sum()/d)
def hd95(a,b,sp):
 if not a.any() or not b.any(): return 0. if a.any()==b.any() else float('inf')
 u=a|b; q=np.argwhere(u); lo=np.maximum(q.min(0)-1,0); hi=np.minimum(q.max(0)+2,u.shape); sl=tuple(slice(int(lo[i]),int(hi[i])) for i in range(3)); a=a[sl]; b=b[sl]; st=ndimage.generate_binary_structure(3,1); sa=a^ndimage.binary_erosion(a,structure=st,border_value=0); sb=b^ndimage.binary_erosion(b,structure=st,border_value=0); da=ndimage.distance_transform_edt(~sa,sampling=tuple(reversed(sp))); db=ndimage.distance_transform_edt(~sb,sampling=tuple(reversed(sp))); return float(np.percentile(np.r_[db[sa],da[sb]],95))
def label(m): return 'inf' if math.isinf(m) else str(int(m))
def choose(rows,key):
 d=max(float(x['mean_dice']) for x in rows); q=[x for x in rows if float(x['mean_dice'])>=d-.001]; h=min(float(x['mean_hd95_mm']) for x in q); q=[x for x in q if abs(float(x['mean_hd95_mm'])-h)<=EPS]; return min(q,key=lambda x:(float(x[key]),float('inf') if x.get('selected_gate_mm',x.get('gate_mm'))=='inf' else float(x.get('selected_gate_mm',x.get('gate_mm')))))
p=argparse.ArgumentParser(); p.add_argument('--kind',choices=('pos','neg'),required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--split-json',type=Path,required=True); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--metrics-root',type=Path,required=True); p.add_argument('--sigmas-mm',type=float,nargs='+',default=SIGMAS); a=p.parse_args(); a.metrics_root.mkdir(parents=True,exist_ok=True)
patients=sorted(json.loads(a.split_json.read_text())['validation'],key=lambda x:num(Path(x))); cases=[]
for sigma in a.sigmas_mm:
 for name in patients:
  case=a.data_root/'train'/name; gt,ref=read(case/'CTV.nii.gz'); n,_=read(case/'nnunet.nii.gz'); item=np.load(a.cache_root/a.kind/f's{int(sigma):02d}mm'/f'{name}.npz'); prob=item['probability'].astype(np.float32); support=item['prompt_mask']>0
  dist=ndimage.distance_transform_edt(~support,sampling=tuple(reversed(ref.GetSpacing()))) if support.any() else None
  for m in GATES:
   allowed=np.ones_like(n,bool) if dist is not None and math.isinf(m) else (np.zeros_like(n,bool) if dist is None else dist<=m)
   pred=n|((prob>=.5)&allowed&~n) if a.kind=='pos' else n&~((prob<.5)&allowed&n)
   cases.append({'kind':a.kind,'patient':name,'sigma_mm':sigma,'gate_mm':label(m),'dice':dice(pred,gt),'hd95_mm':hd95(pred,gt,ref.GetSpacing()),'prompt_support_voxels':int(support.sum())})
with (a.metrics_root/f'validation_{a.kind}_per_case.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=cases[0]); w.writeheader(); w.writerows(cases)
grid=[]
for s in a.sigmas_mm:
 for m in GATES:
  q=[x for x in cases if x['sigma_mm']==s and x['gate_mm']==label(m)]; grid.append({'kind':a.kind,'sigma_mm':s,'gate_mm':label(m),'mean_dice':float(np.mean([x['dice'] for x in q])),'mean_hd95_mm':float(np.mean([x['hd95_mm'] for x in q])),'cases':len(q)})
with (a.metrics_root/f'validation_{a.kind}_grid.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=grid[0]); w.writeheader(); w.writerows(grid)
profile=[]
for s in a.sigmas_mm:
 q=choose([x for x in grid if x['sigma_mm']==s],'gate_mm'); q={**q,'selected_gate_mm':q['gate_mm']}; profile.append(q)
with (a.metrics_root/f'validation_{a.kind}_profile.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=profile[0]); w.writeheader(); w.writerows(profile)
locked=choose(profile,'sigma_mm'); (a.metrics_root/f'LOCKED_{a.kind.upper()}_VALIDATION_CONFIG.json').write_text(json.dumps({'selection_order':'epoch -> m*(sigma) -> sigma*','dice_equivalence':.001,'locked':locked,'per_sigma_best':profile},indent=2)); print(json.dumps(locked,indent=2))
