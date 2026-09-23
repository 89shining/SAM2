#!/usr/bin/env python3
"""Final, test-only POS+NEG skeleton fusion using validation-locked settings."""
from __future__ import annotations
import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

def dice(a,b):
    d=int(a.sum()+b.sum()); return 1.0 if d==0 else float(2*np.logical_and(a,b).sum()/d)
def hd95(a,b,sp):
    if not a.any() or not b.any(): return 0.0 if a.any()==b.any() else float('inf')
    u=a|b; pts=np.argwhere(u); lo=np.maximum(pts.min(0)-1,0); hi=np.minimum(pts.max(0)+2,u.shape); crop=tuple(slice(int(lo[i]),int(hi[i])) for i in range(3)); a=a[crop]; b=b[crop]
    st=ndimage.generate_binary_structure(3,1); sa=a^ndimage.binary_erosion(a,structure=st,border_value=0); sb=b^ndimage.binary_erosion(b,structure=st,border_value=0); sampling=tuple(reversed(sp)); da=ndimage.distance_transform_edt(~sa,sampling=sampling); db=ndimage.distance_transform_edt(~sb,sampling=sampling); return float(np.percentile(np.r_[db[sa],da[sb]],95))
def gate(prob,prompt,n,kind,m):
    if not prompt.any(): return n.copy()
    d=None if math.isinf(m) else ndimage.distance_transform_edt(~prompt,sampling=None) # overwritten in caller
    raise RuntimeError('internal')
def gated(prob,prompt,n,kind,m,spacing):
    if not prompt.any(): return n.copy()
    allowed=np.ones_like(prompt,bool) if math.isinf(m) else ndimage.distance_transform_edt(~prompt,sampling=tuple(reversed(spacing)))<=m
    if kind=='pos': return n | prompt | ((prob>=.5)&allowed&~n&~prompt)
    return n & ~(prompt | ((prob<.5)&allowed&n&~prompt))
def write(mask,ref,path):
    im=sitk.GetImageFromArray(mask.astype(np.uint8)); im.CopyInformation(ref); sitk.WriteImage(im,str(path),True)

p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--selection-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--fractions',type=float,nargs='+',default=[0,.25,.5,.75,1]); a=p.parse_args()
locks={k:json.loads((a.selection_root/f'LOCKED_{k.upper()}_VALIDATION_CONFIG.json').read_text())['locked'] for k in ('pos','neg')}
a.output_root.mkdir(parents=True,exist_ok=True); rows=[]
for frac in a.fractions:
    tag=f'p{int(round(frac*100)):03d}';
    for case in sorted((a.data_root/'test').glob('p_*'), key=lambda q:int(re.search(r'(\d+)$',q.name).group(1))):
        ref=sitk.ReadImage(str(case/'image.nii.gz')); gt=sitk.GetArrayFromImage(sitk.ReadImage(str(case/'CTV.nii.gz')))>0; n=sitk.GetArrayFromImage(sitk.ReadImage(str(case/'nnunet.nii.gz')))>0
        branch={}; prompts={}
        for k in ('pos','neg'):
            t=int(float(locks[k]['thickness_mm'])); m=float('inf') if locks[k]['selected_gate_mm']=='inf' else float(locks[k]['selected_gate_mm'])
            item=np.load(a.cache_root/f'fraction_{tag}'/k/f't{t:02d}mm'/f'{case.name}.npz'); prompts[k]=item['prompt_mask'].astype(bool); branch[k]=gated(item['probability'].astype(np.float32),prompts[k],n,k,m,ref.GetSpacing())
        # Same collision algebra for Direct and final fusion.
        direct=(n|prompts['pos']) & ~(n&prompts['neg'])
        fused=branch['pos'] & ~(n & ~branch['neg'])
        out=a.output_root/tag/case.name; out.mkdir(parents=True,exist_ok=True)
        for name,mask in [('POS_pred.nii.gz',branch['pos']),('NEG_pred.nii.gz',branch['neg']),('Direct_pred.nii.gz',direct),('CTV_pred.nii.gz',fused)]: write(mask,ref,out/name)
        for name,mask in [('nnunet',n),('direct',direct),('pos_only',branch['pos']),('neg_only',branch['neg']),('fusion',fused)]: rows.append({'fraction':frac,'patient':case.name,'method':name,'dice':dice(mask,gt),'hd95_mm':hd95(mask,gt,ref.GetSpacing())})
with (a.output_root/'fusion_test_per_case.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
summary=[]
for frac in a.fractions:
    for method in ('nnunet','direct','pos_only','neg_only','fusion'):
        x=[r for r in rows if r['fraction']==frac and r['method']==method]; summary.append({'fraction':frac,'method':method,'cases':len(x),'mean_dice':float(np.mean([r['dice'] for r in x])),'mean_hd95_mm':float(np.mean([r['hd95_mm'] for r in x]))})
with (a.output_root/'fusion_test_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
(a.output_root/'VALIDATION_LOCKS_USED.json').write_text(json.dumps(locks,indent=2)); print(json.dumps(summary,indent=2))
