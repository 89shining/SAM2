#!/usr/bin/env python3
"""Fuse final validation-locked Gaussian POS and NEG models on test only."""
from __future__ import annotations
import argparse,csv,json,math,re
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
def dice(a,b):
 d=int(a.sum()+b.sum()); return 1. if d==0 else float(2*(a&b).sum()/d)
def hd(a,b,sp):
 if not a.any() or not b.any(): return 0. if a.any()==b.any() else float('inf')
 u=a|b; q=np.argwhere(u); lo=np.maximum(q.min(0)-1,0); hi=np.minimum(q.max(0)+2,u.shape); sl=tuple(slice(int(lo[i]),int(hi[i])) for i in range(3)); a=a[sl];b=b[sl];st=ndimage.generate_binary_structure(3,1);sa=a^ndimage.binary_erosion(a,structure=st,border_value=0);sb=b^ndimage.binary_erosion(b,structure=st,border_value=0);da=ndimage.distance_transform_edt(~sa,sampling=tuple(reversed(sp)));db=ndimage.distance_transform_edt(~sb,sampling=tuple(reversed(sp)));return float(np.percentile(np.r_[db[sa],da[sb]],95))
def gate(prob,support,n,kind,m,sp):
 allowed=np.zeros_like(n,bool) if not support.any() else (np.ones_like(n,bool) if math.isinf(m) else ndimage.distance_transform_edt(~support,sampling=tuple(reversed(sp)))<=m)
 return n|((prob>=.5)&allowed&~n) if kind=='pos' else n&~((prob<.5)&allowed&n)
def save(a,ref,p): im=sitk.GetImageFromArray(a.astype(np.uint8));im.CopyInformation(ref);sitk.WriteImage(im,str(p),True)
p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--selection-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--fractions',type=float,nargs='+',default=[0,.25,.5,.75,1]);a=p.parse_args(); locks={k:json.loads((a.selection_root/f'LOCKED_{k.upper()}_VALIDATION_CONFIG.json').read_text())['locked'] for k in ('pos','neg')};rows=[]
for f in a.fractions:
 tag=f'p{int(round(f*100)):03d}'
 for case in sorted((a.data_root/'test').glob('p_*'),key=lambda x:int(re.search(r'(\d+)$',x.name).group(1))):
  ref=sitk.ReadImage(str(case/'image.nii.gz'));gt=sitk.GetArrayFromImage(sitk.ReadImage(str(case/'CTV.nii.gz')))>0;n=sitk.GetArrayFromImage(sitk.ReadImage(str(case/'nnunet.nii.gz')))>0;pred={}
  for k in ('pos','neg'):
   l=locks[k];s=int(float(l['sigma_mm']));m=float('inf') if l['selected_gate_mm']=='inf' else float(l['selected_gate_mm']);x=np.load(a.cache_root/f'fraction_{tag}'/k/f's{s:02d}mm'/f'{case.name}.npz');pred[k]=gate(x['probability'].astype(np.float32),x['prompt_mask']>0,n,k,m,ref.GetSpacing())
  fused=pred['pos']&~(n&~pred['neg']);out=a.output_root/tag/case.name;out.mkdir(parents=True,exist_ok=True)
  for name,x in [('POS_pred.nii.gz',pred['pos']),('NEG_pred.nii.gz',pred['neg']),('CTV_pred.nii.gz',fused)]:save(x,ref,out/name)
  for method,x in [('nnunet',n),('pos_only',pred['pos']),('neg_only',pred['neg']),('fusion',fused)]:rows.append({'fraction':f,'patient':case.name,'method':method,'dice':dice(x,gt),'hd95_mm':hd(x,gt,ref.GetSpacing())})
with (a.output_root/'per_case.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for f in a.fractions:
 for m in ('nnunet','pos_only','neg_only','fusion'):
  q=[x for x in rows if x['fraction']==f and x['method']==m];summary.append({'fraction':f,'method':m,'cases':len(q),'mean_dice':float(np.mean([x['dice'] for x in q])),'mean_hd95_mm':float(np.mean([x['hd95_mm'] for x in q]))})
with (a.output_root/'summary.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
(a.output_root/'validation_locks.json').write_text(json.dumps(locks,indent=2));print(json.dumps(summary,indent=2))
