#!/usr/bin/env python
"""Frozen-plan, held-out-fold Stage-1 SAM2 evaluation (no Stage-2 logic)."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

STAGE1 = Path(__file__).resolve().parents[1] / 'Stage1-mask'
sys.path.insert(0, str(STAGE1))
from io_utils import (DEFAULT_DATA_ROOT, DEFAULT_INIT_CKPT, DEFAULT_MODEL_CFG, DEFAULT_SPLIT_PATH,
                      RectalCTVVolumeDataset, build_model, list_patient_dirs, make_or_load_splits,
                      patient_id_from_dir)
from bidirectional_tracking import bidirectional_outputs
from experiment_core import hd95_asd, unprompted_slice_3d_dsc
from training.utils.data_utils import collate_fn

TRAIN = Path('/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults')
PLAN = Path('/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/TestResults/prompt_plans/stage1_prompt_plan.json')

def args_():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT/'train'); p.add_argument('--split-path',type=Path,default=DEFAULT_SPLIT_PATH); p.add_argument('--model-cfg',default=DEFAULT_MODEL_CFG); p.add_argument('--init-ckpt',type=Path,default=DEFAULT_INIT_CKPT); p.add_argument('--train-results',type=Path,default=TRAIN); p.add_argument('--checkpoint',type=Path,help='Explicit checkpoint override, e.g. a Stage2 fold0 best.pth.'); p.add_argument('--plan',type=Path,default=PLAN); p.add_argument('--output-dir',type=Path); p.add_argument('--fold',type=int,default=0); p.add_argument('--external-test',action='store_true',help='Evaluate every case in --data-root with the requested checkpoint; do not apply a training split.'); p.add_argument('--allow-legacy-fold0-checkpoint',action='store_true'); p.add_argument('--device',default='cuda'); p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--save-predictions',action='store_true'); p.add_argument('--lora-r',type=int,default=4); p.add_argument('--lora-alpha',type=int,default=16); p.add_argument('--lora-dropout',type=float,default=.1); p.add_argument('--input-size',type=int,default=512); p.add_argument('--image-encoder-activation-checkpointing',action=argparse.BooleanOptionalAction,default=True); return p.parse_args()

def dice(pred,gt):
 d=pred.sum()+gt.sum(); return 1.0 if d==0 else float((2*(pred & gt).sum()+1e-6)/(d+1e-6))

def endpoint_mae(pred,gt,reference):
 if not gt.any(): raise RuntimeError('Ground-truth CTV is empty')
 if not pred.any(): return None,None
 # Array axis 0 is image z. Convert it to physical superior coordinates so
 # Sup/Inf retain their anatomical meaning even if image z is reversed.
 origin_z, step_z = reference.GetOrigin()[2], reference.GetDirection()[8] * reference.GetSpacing()[2]
 def ends(mask):
  coords=origin_z + step_z*np.where(mask)[0]; return float(coords.max()),float(coords.min())
 ps,pi=ends(pred); gs,gi=ends(gt); return abs(ps-gs),abs(pi-gi)

def batch_for(case,device):
 ds=RectalCTVVolumeDataset([case]); return collate_fn([ds[0]],dict_key='eso_ctv_k1_5').to(device,non_blocking=True)

def save_prediction(pred,case,dest):
 ref=sitk.ReadImage(str(case/'CTV.nii.gz')); out=sitk.GetImageFromArray(pred.astype(np.uint8)); out.CopyInformation(ref); dest.parent.mkdir(parents=True,exist_ok=True); sitk.WriteImage(out,str(dest),True)

def describe(values):
 values=np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype=float)
 if not len(values): return {'n_valid':0,'mean':None,'std':None,'median':None,'q25':None,'q75':None}
 return {'n_valid':int(len(values)),'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(values)>1 else 0.0,'median':float(np.median(values)),'q25':float(np.quantile(values,.25)),'q75':float(np.quantile(values,.75))}

def main():
 a=args_(); device=torch.device(a.device); document=json.loads(a.plan.read_text(encoding='utf-8'))
 if document.get('schema_version')!=2 or document.get('plan_type') not in {'deterministic','random_robustness'} or not isinstance(document.get('patients'),dict): raise ValueError('Plan must be a schema_version=2 immutable plan document')
 plan=document['patients']; mode='main_deterministic' if document['plan_type']=='deterministic' else 'random_robustness'; out=a.output_dir or (a.train_results.parent/'TestResults'/('fold0_checkpoint' if a.external_test else mode/f'fold_{a.fold}_only')); out.mkdir(parents=True,exist_ok=True)
 cases={patient_id_from_dir(x):x for x in list_patient_dirs(a.data_root)}
 if a.external_test:
  if a.fold != 0: raise ValueError('External test mode is restricted to the available fold0 checkpoint')
  fold_of={pid:a.fold for pid in cases}; heldout=set(cases)
  if not heldout or not heldout.issubset(set(map(int,plan))): raise RuntimeError('Frozen plan/data coverage mismatch for external test')
 else:
  if not a.split_path.is_file(): raise FileNotFoundError(f'Frozen split file not found: {a.split_path}')
  folds=make_or_load_splits(list(cases.values()),5,20260909,a.split_path); fold_of={patient_id_from_dir(p):int(f['fold']) for f in folds for p in f['val']}; heldout={pid for pid,value in fold_of.items() if value==a.fold}
  if a.fold not in range(5) or set(fold_of)!=set(cases) or not heldout or not heldout.issubset(set(map(int,plan))): raise RuntimeError('Frozen plan/data/fold coverage mismatch for requested fold')
 grouped=defaultdict(list)
 for pid_s,patient in plan.items():
  if int(pid_s) not in heldout: continue
  for key,record in patient['plans'].items():
   if not record['feasible']: raise RuntimeError(f'infeasible plan p_{pid_s} {key}')
   grouped[fold_of[int(pid_s)],key].append((int(pid_s),[int(z) for z in record['slices']]))
 rows=[]
 for fold in [a.fold]:
  ckpt=a.checkpoint or a.train_results/'Mixed_K1_5'/f'fold_{fold}'/'checkpoints'/'best.pth'
  if not ckpt.is_file(): raise FileNotFoundError(ckpt)
  saved=torch.load(ckpt,map_location='cpu',weights_only=False)
  legacy=not isinstance(saved.get('args'),dict) or 'fold' not in saved['args']
  if legacy:
   if not (a.allow_legacy_fold0_checkpoint and fold==0): raise RuntimeError(f'checkpoint lacks frozen training fold: {ckpt}')
  else:
   if int(saved['args']['fold'])!=fold: raise RuntimeError(f'checkpoint fold mismatch: {ckpt}')
   for name in ('model_cfg','input_size','lora_r','lora_alpha','lora_dropout'):
    if name not in saved['args'] or str(saved['args'][name])!=str(getattr(a,name)): raise RuntimeError(f'checkpoint/config mismatch for {name}: {ckpt}')
  model,_=build_model(a.model_cfg,a.init_ckpt,device,a); model.load_state_dict(saved['model'],strict=True); model.eval()
  for (group_fold,key),tasks in sorted(grouped.items()):
   if group_fold!=fold: continue
   for pid,prompts in sorted(tasks):
    case=cases[pid]; batch=batch_for(case,device); gt=batch.masks[:,0].detach().cpu().numpy().astype(bool); positive=np.flatnonzero(gt.reshape(gt.shape[0],-1).any(1));
    if not set(prompts).issubset(set(map(int,positive))) or any(b-a < 2 for a,b in zip(prompts,prompts[1:])): raise RuntimeError(f'Invalid frozen prompts p_{pid} {key}')
    with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=a.amp and device.type=='cuda'):
     base=model.forward_image(batch.flat_img_batch); outputs=bidirectional_outputs(model,batch,prompts,False,base,gt_masks_for_track_step=False)
    pred=torch.stack([x['pred_masks_high_res'][:,0] for x in outputs]).squeeze(1).gt(0).cpu().numpy(); ref=sitk.ReadImage(str(case/'CTV.nii.gz')); prediction_empty=not pred.any()
    if not gt.any(): raise RuntimeError(f'Ground-truth CTV is empty: p_{pid}')
    if prediction_empty: hd95=sup=inf=None
    else:
     spacing=tuple(reversed(ref.GetSpacing())); hd95,_=hd95_asd(pred,gt,spacing); sup,inf=endpoint_mae(pred,gt,ref)
    rows.append({'patient_id':pid,'fold':fold,'plan_key':key,'K':len(prompts),'prompt_slices':json.dumps(prompts),'whole_volume_dice':dice(pred,gt),'unprompted_dice':unprompted_slice_3d_dsc(outputs,batch.masks,prompts),'hd95_3d_mm':hd95,'prediction_empty':int(prediction_empty),'sup_endpoint_mae_mm':sup,'inf_endpoint_mae_mm':inf})
    if a.save_predictions: save_prediction(pred,case,out/'predictions'/key/f'p_{pid:03d}.nii.gz')
    print(f'[fold{fold} {key}] p_{pid:03d} Dice={rows[-1]["whole_volume_dice"]:.4f}',flush=True)
  del model
 with (out/'per_case_metrics.csv').open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 metrics=['whole_volume_dice','hd95_3d_mm','sup_endpoint_mae_mm','inf_endpoint_mae_mm','unprompted_dice']; summary={'evaluation_scope':'external_test_with_fold0_checkpoint' if a.external_test else f'fold_{a.fold}_only','checkpoint':str(ckpt),'checkpoint_identity':'legacy_path_scoped_fold0' if a.allow_legacy_fold0_checkpoint else 'checkpoint_metadata_verified','primary_metrics':metrics[:4],'auxiliary_metrics':['unprompted_dice'],'plan':str(a.plan),'plan_type':document['plan_type'],'n':len(rows),'by_plan':{},'by_K':{}}
 for key in sorted({r['plan_key'] for r in rows}):
  selected=[r for r in rows if r['plan_key']==key]; summary['by_plan'][key]={'n':len(selected),'n_empty_prediction':sum(r['prediction_empty'] for r in selected),**{m:describe([r[m] for r in selected]) for m in metrics}}
 for k in range(1,6):
  selected=[r for r in rows if r['K']==k]; summary['by_K'][f'K{k}']={'n':len(selected),'n_empty_prediction':sum(r['prediction_empty'] for r in selected),**{m:describe([r[m] for r in selected]) for m in metrics}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
