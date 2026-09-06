from __future__ import annotations
import argparse,csv,json,torch,torch.nn.functional as F,SimpleITK as sitk,numpy as np
from pathlib import Path
from config import *
from data import CTVGTVCase,patient_dirs
from model import build_singlegtv
from engine import episode,midpoint_bidirectional
from metrics import dice,hd95,asd,boundary_metrics
def collate(v):
 from training.utils.data_utils import collate_fn
 return collate_fn(v,dict_key='fullvolume_singlegtv_ctv')
def materialize(m,c,d):
 b,p=c.video(list(range(min(8,len(c.image)))));b=collate([b]).to(d)
 with torch.no_grad():episode(m,b,p,list(range(min(8,len(c.image)))))
def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--out',type=Path,default=RESULT_ROOT/'TestResults');a.add_argument('--device',default='cuda');x=a.parse_args();x.out.mkdir(parents=True,exist_ok=True);dev=torch.device(x.device);paths={p.name:p for p in patient_dirs(DATA_ROOT,'test')};mapping=sorted([r for r in csv.DictReader((DATA_ROOT/'patient_id_mapping.csv').open()) if r['split']=='test'],key=lambda r:int(r['new_id'][2:]));first=CTVGTVCase(paths[mapping[0]['new_id']]);m,n=build_singlegtv(SAM2_ROOT,MODEL_CFG,CHECKPOINT,dev,LORA_RANK,LORA_ALPHA,LORA_DROPOUT);materialize(m,first,dev);m.load_state_dict(torch.load(x.checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);m.eval();rows=[]
 for r in mapping:
  c=CTVGTVCase(paths[r['new_id']]);b,p=c.video(list(range(len(c.image))));b=collate([b]).to(dev);q=torch.cat([midpoint_bidirectional(m,b,p)[i] for i in range(len(c.image))]).sigmoid();q=F.interpolate(q,size=c.ctv.shape[-2:],mode='bilinear',align_corners=False)[:,0].cpu().numpy().astype('float32');o=c.restore_probability_to_original(q)>.5;g=sitk.GetArrayFromImage(c.original_ctv)>0;name=f"EsoCTV_test_{r['new_id']}.nii";im=sitk.GetImageFromArray(o.astype('uint8'));im.CopyInformation(c.original_ctv);sitk.WriteImage(im,str(x.out/name),True);z={'patient':r['new_id'],'original_id':r['original_id'],'prediction_file':name,'dice':dice(o,g),'hd95_mm':hd95(o,g,c.spacing_zyx),'asd_mm':asd(o,g,c.spacing_zyx)};z.update(boundary_metrics(o,g,c.spacing_zyx[0]));rows.append(z)
 with (x.out/'test_patient_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (x.out/'test_summary.json').write_text(json.dumps({'n_cases':25,'metrics_mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k],(float,int))}},indent=2))
if __name__=='__main__':main()