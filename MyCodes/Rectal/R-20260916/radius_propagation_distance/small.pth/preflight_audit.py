#!/usr/bin/env python3
"""Fail-closed preflight audit for R-20260916 before any training starts."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

RADII=(2,4,6,8,10,12,14,16,18,20)
EXPECTED_MD5="2933e8e1ef887ecc62606ffa3c5a3eb0"

def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()

def arr(p: Path):
    im=sitk.ReadImage(str(p)); return sitk.GetArrayFromImage(im)>0, im

def expected_disk(postprocessed: np.ndarray, spacing_xyz, radius: float) -> np.ndarray:
    """Independent voxelwise reconstruction: no component/raw clipping allowed."""
    from scipy import ndimage
    sx,sy,_=spacing_xyz; out=np.zeros_like(postprocessed, dtype=bool)
    for z in range(postprocessed.shape[0]):
        labels,n=ndimage.label(postprocessed[z], structure=np.ones((3,3),bool))
        for label in range(1,n+1):
            component=labels==label
            dist=ndimage.distance_transform_edt(component, sampling=(sy,sx))
            cy,cx=np.unravel_index(np.argmax(dist),dist.shape)
            yy,xx=np.ogrid[:component.shape[0],:component.shape[1]]
            out[z] |= ((xx-cx)*sx)**2+((yy-cy)*sy)**2 <= radius**2
    return out

def main() -> None:
    import argparse
    q=argparse.ArgumentParser()
    q.add_argument('--data-root',type=Path,required=True); q.add_argument('--split-json',type=Path,required=True)
    q.add_argument('--output-root',type=Path,required=True); q.add_argument('--code-root',type=Path,required=True)
    a=q.parse_args()
    pass_marker=a.output_root/'PREFLIGHT_PASS'
    pass_marker.unlink(missing_ok=True)
    split=json.loads(a.split_json.read_text())
    checks=[]
    def check(name, ok, detail): checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail})
    test_cases=sorted((a.data_root/'test').glob('p_*'))
    check('split_counts',len(split['train'])==99 and len(split['validation'])==11 and len(test_cases)==36, str({'train':len(split['train']),'validation':len(split['validation']),'test_directory':len(test_cases)}))
    check('split_md5',md5(a.split_json)==EXPECTED_MD5,md5(a.split_json))
    cfg=json.loads((a.code_root/'experiment_config.json').read_text())
    check('locked_config',cfg['training']['input_size']==512 and cfg['training']['window_slices']==9 and cfg['training']['epochs_max']==60 and cfg['training']['early_stopping_patience']==10,str(cfg['training']))
    rows=[]
    for subset,names in (('train',split['train']+split['validation']),('test',[p.name for p in test_cases])):
      for patient in names:
       case=a.data_root/subset/patient
       _,ref=arr(case/'image.nii.gz')
       for branch in ('pos','neg'):
        raw,_=arr(case/f'{branch}_raw.nii.gz')
        postprocessed,_=arr(case/f'{branch}_prompt_erode2_top3_area50.nii.gz')
        centers=[]; statuses=[]; available_slices=[]; disks=[]
        for r in RADII:
          disk,dref=arr(case/f'{branch}_prompt_disk{r}mm.nii.gz')
          same_geo=(dref.GetSize()==ref.GetSize() and dref.GetSpacing()==ref.GetSpacing() and dref.GetOrigin()==ref.GetOrigin() and dref.GetDirection()==ref.GetDirection())
          outside=int((disk & ~raw).sum())
          expected=expected_disk(postprocessed, ref.GetSpacing(), r)
          statuses.append(bool(disk.any()) if raw.any() else not disk.any())
          centers.append(int(disk.sum()))
          available_slices.append(tuple(np.flatnonzero(disk.reshape(disk.shape[0],-1).any(axis=1)).tolist()))
          disks.append(disk)
          rows.append({'subset':subset,'patient':patient,'branch':branch,'radius_mm':r,'disk_voxels':int(disk.sum()),'outside_raw_voxels':outside,'geometry_matches_image':same_geo,'matches_expected_unclipped_disk':bool(np.array_equal(disk,expected))})
        check(f'{subset}/{patient}/{branch}/files_geometry',all(row['geometry_matches_image'] for row in rows[-len(RADII):]),'all radii')
        check(f'{subset}/{patient}/{branch}/expected_unclipped_disk',all(row['matches_expected_unclipped_disk'] for row in rows[-len(RADII):]),'saved disk equals independent physical reconstruction')
        check(f'{subset}/{patient}/{branch}/same_valid_slices',all(x==available_slices[0] for x in available_slices),'all radii share disk-bearing slices/centers')
        check(f'{subset}/{patient}/{branch}/nested_disks',all(np.all(disks[i] <= disks[i+1]) for i in range(len(disks)-1)),'radii are monotonic from the same centers')
    metrics=a.output_root/'metrics'; metrics.mkdir(parents=True,exist_ok=True)
    with (metrics/'preflight_prompt_audit.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pos=Path(cfg['initialization']['pos']); neg=Path(cfg['initialization']['neg'])
    hashes={'pos_0720_best_sha256':hashlib.sha256(pos.read_bytes()).hexdigest(),'neg_0720_best_sha256':hashlib.sha256(neg.read_bytes()).hexdigest()}
    (metrics/'initial_checkpoint_hash.txt').write_text('\n'.join(f'{k} {v}' for k,v in hashes.items())+'\n')
    check('warm_starts_exist',pos.is_file() and neg.is_file(),'0720 POS and NEG best.pth')
    with (metrics/'preflight_checks.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=('check','status','detail'));w.writeheader();w.writerows(checks)
    failures=[x for x in checks if x['status']=='FAIL']
    if failures:
      print(json.dumps(failures,indent=2)); sys.exit(1)
    pass_marker.touch(); print('PREFLIGHT_PASS')
if __name__=='__main__': main()
