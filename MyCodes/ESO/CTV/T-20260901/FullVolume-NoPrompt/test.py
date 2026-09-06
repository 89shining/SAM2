from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, SimpleITK as sitk
from config import *
from data import CTVCase,patient_dirs,load_splits
from model import build_fullvolume_no_prompt
from engine import midpoint_bidirectional
from metrics import dice,hd95,asd,boundary_metrics

def collate(v):
 from training.utils.data_utils import collate_fn
 return collate_fn(v,dict_key='fullvolume_no_prompt_ctv')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows_write(p,rows):
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,default=RESULT_ROOT/'TestResults');ap.add_argument('--device',default='cuda');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);core,_=build_fullvolume_no_prompt(SAM2_ROOT,MODEL_CFG,CHECKPOINT,torch.device(a.device),LORA_RANK,LORA_ALPHA,LORA_DROPOUT);core.load_state_dict(ck['model'],strict=True);core.eval()
 mapping=list(csv.DictReader((DATA_ROOT/'patient_id_mapping.csv').open())); testmap=sorted([r for r in mapping if r['split']=='test'],key=lambda r:int(r['new_id'][2:])); paths={p.name:p for p in patient_dirs(DATA_ROOT,'test')};rows=[]
 for r in testmap:
  pid=r['new_id'];case=CTVCase(paths[pid]);b=collate([case.video(range(len(case.image)))]).to(a.device);preds=midpoint_bidirectional(core,b);prob=torch.cat([preds[i] for i in range(len(preds))]).sigmoid();prob=F.interpolate(prob,size=case.ctv.shape[-2:],mode='bilinear',align_corners=False);probability=prob[:,0].cpu().numpy().astype(np.float32);original_probability=case.restore_probability_to_original(probability);original=original_probability>.5;gt=sitk.GetArrayFromImage(case.original_ctv)>0;fname=f'EsoCTV_test_{pid}.nii';im=sitk.GetImageFromArray(original.astype(np.uint8));im.CopyInformation(case.original_ctv);sitk.WriteImage(im,str(a.out/fname),True)
  row={'patient':pid,'original_id':r['original_id'],'prediction_file':fname,'dice':dice(original,gt),'hd95_mm':hd95(original,gt,case.spacing_zyx),'asd_mm':asd(original,gt,case.spacing_zyx)};row.update(boundary_metrics(original,gt,case.spacing_zyx[0]));rows.append(row)
 rows_write(a.out/'test_patient_metrics.csv',rows);means={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k],(float,int)) and k not in ('gt_superior_slice','gt_inferior_slice','pred_superior_slice','pred_inferior_slice')};(a.out/'test_summary.json').write_text(json.dumps({'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'metrics_mean':means,'n_cases':len(rows)},indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()