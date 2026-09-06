from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, SimpleITK as sitk
from config import DATA_ROOT
from data import patient_dirs
from metrics import dice,hd95,asd,boundary_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--pred-dir',type=Path,required=True);a=p.parse_args();mapping=list(csv.DictReader((DATA_ROOT/'patient_id_mapping.csv').open()));testmap=sorted([r for r in mapping if r['split']=='test'],key=lambda r:int(r['new_id'][2:]));paths={x.name:x for x in patient_dirs(DATA_ROOT,'test')};rows=[]
 for r in testmap:
  pid=r['new_id']; pred_path=a.pred_dir/f'EsoCTV_test_{pid}.nii'
  if not pred_path.exists():raise FileNotFoundError(pred_path)
  ref=sitk.ReadImage(str(paths[pid]/'CTV.nii.gz')); pred=sitk.ReadImage(str(pred_path))
  if pred.GetSize()!=ref.GetSize() or pred.GetSpacing()!=ref.GetSpacing() or pred.GetOrigin()!=ref.GetOrigin() or pred.GetDirection()!=ref.GetDirection():raise RuntimeError(f'{pid}: prediction not restored to original CTV geometry')
  pa=sitk.GetArrayFromImage(pred)>0;ga=sitk.GetArrayFromImage(ref)>0;spacing=tuple(reversed(ref.GetSpacing()));row={'patient':pid,'original_id':r['original_id'],'dice':dice(pa,ga),'hd95_mm':hd95(pa,ga,spacing),'asd_mm':asd(pa,ga,spacing)};row.update(boundary_metrics(pa,ga,spacing[0]));rows.append(row)
 with (a.pred_dir/'test_patient_metrics_recomputed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 mean={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k],float)};(a.pred_dir/'test_summary_recomputed.json').write_text(json.dumps({'n_cases':len(rows),'metrics_mean':mean},indent=2));print(json.dumps(mean,indent=2))
if __name__=='__main__':main()