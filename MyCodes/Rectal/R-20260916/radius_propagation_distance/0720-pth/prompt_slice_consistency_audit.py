#!/usr/bin/env python3
"""Assert deterministic 50% selected prompt layers are identical across radii."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
RADII=(2,4,6,8,10,12,14,16,18,20)
def selected(mask):
    available=np.flatnonzero(mask.reshape(mask.shape[0],-1).any(axis=1))
    if not len(available): return np.array([],dtype=int)
    n=max(1,int(np.ceil(len(available)*.5)))
    return available[np.unique(np.rint(np.linspace(0,len(available)-1,n)).astype(int))]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--split-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 pass_marker=a.output.parent.parent/'PROMPT_SLICE_CONSISTENCY_PASS'
 pass_marker.unlink(missing_ok=True)
 s=json.loads(a.split_json.read_text()); rows=[]; bad=[]
 test_names=[p.name for p in sorted((a.data_root/'test').glob('p_*'))]
 for subset,names in (('train',s['train']+s['validation']),('test',test_names)):
  for patient in names:
   for branch in ('pos','neg'):
    reference=None
    for r in RADII:
     im=sitk.ReadImage(str(a.data_root/subset/patient/f'{branch}_prompt_disk{r}mm.nii.gz')); m=sitk.GetArrayFromImage(im)>0; z=selected(m)
     if reference is None: reference=z
     ok=np.array_equal(z,reference)
     rows.append({'subset':subset,'patient':patient,'branch':branch,'radius':r,'num_available_slices':int(m.reshape(m.shape[0],-1).any(1).sum()),'num_selected_slices':len(z),'selected_slice_indices':','.join(map(str,z)),'matches_reference_radius':ok})
     if not ok: bad.append(rows[-1])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 if bad: raise SystemExit(f'cross-radius deterministic slice audit failed: {bad[:3]}')
 pass_marker.touch()
 print('PROMPT_SLICE_CONSISTENCY_PASS')
if __name__=='__main__':main()
