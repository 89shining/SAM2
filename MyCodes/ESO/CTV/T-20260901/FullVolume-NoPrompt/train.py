from __future__ import annotations
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from config import *
from data import CTVCase,patient_dirs,load_splits,sample_epoch_plans
from model import build_fullvolume_no_prompt
from engine import episode,midpoint_bidirectional
from metrics import dice,hd95,asd,boundary_metrics

def collate(v):
 from training.utils.data_utils import collate_fn
 return collate_fn(v,dict_key='fullvolume_no_prompt_ctv')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_all(seed): random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def capture_rng(): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
 if torch.cuda.is_available() and s.get('cuda') is not None: torch.cuda.set_rng_state_all(s['cuda'])
def _probability_resampled(case,preds):
 prob=torch.cat([preds[i] for i in range(len(preds))]).sigmoid();prob=F.interpolate(prob,size=case.ctv.shape[-2:],mode='bilinear',align_corners=False)
 return prob[:,0].cpu().numpy().astype(np.float32)
def write_rows(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def append_row(path,row):
 exists=Path(path).exists()
 with Path(path).open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(row));
  if not exists:w.writeheader()
  w.writerow(row)
def read_history(path):
 if not Path(path).exists():return []
 return list(csv.DictReader(Path(path).open()))
def plot_history(path,out):
 try:
  import matplotlib.pyplot as plt
  h=read_history(path)
  if not h:return
  e=[int(r['epoch']) for r in h];plt.figure(figsize=(8,5));plt.plot(e,[float(r['train_loss']) for r in h],label='train loss');plt.plot(e,[float(r['validation_loss']) for r in h],label='validation loss');plt.plot(e,[float(r['validation_dice_original_space']) for r in h],label='validation Dice3D');plt.xlabel('epoch');plt.legend();plt.tight_layout();plt.savefig(Path(out)/'training_curve.png',dpi=160);plt.close()
 except Exception as exc:Path(out,'training_curve_warning.txt').write_text(str(exc))
def full_validate(core,cases,device):
 rows=[];core.eval()
 for c in cases.values():
  b=collate([c.video(range(len(c.image)))]).to(device); probability=_probability_resampled(c,midpoint_bidirectional(core,b));original_probability=c.restore_probability_to_original(probability);original=original_probability>0.5;gt=(__import__('SimpleITK').GetArrayFromImage(c.original_ctv)>0)
  row={'patient':c.patient,'dice':dice(original,gt),'hd95_mm':hd95(original,gt,c.spacing_zyx),'asd_mm':asd(original,gt,c.spacing_zyx)};row.update(boundary_metrics(original,gt,c.spacing_zyx[0]));rows.append(row)
 return rows
def expected_config(args,fold,train,valid):
 return {'experiment':'Independent FullVolume-NoPrompt','target':'CTV','no_si_range':True,'no_gtv':True,'no_spatial_prompt':True,'sam2_checkpoint':str(CHECKPOINT),'sam2_checkpoint_sha256':sha(CHECKPOINT),'sam2_config':MODEL_CFG,'split_path':str(SPLIT_PATH),'split_sha256':sha(SPLIT_PATH),'fold':args.fold,'lambda_rank':args.lambda_rank,'train_ids':sorted(train),'validation_ids':sorted(valid),'target_spacing_xyz':TARGET_SPACING_XYZ,'image_size':IMAGE_SIZE,'max_epochs':args.epochs,'patience':args.patience,'early_stopping':'original_space_mean_validation_dice','lora':{'r':LORA_RANK,'alpha':LORA_ALPHA,'dropout':LORA_DROPOUT,'target':'image_encoder fused qkv Q/V'},'trainable_profile':'image LoRA; memory encoder; memory attention; mask decoder'}
def assert_or_write_config(path,meta,resume):
 if path.exists():
  old=json.loads(path.read_text());keys=('experiment','split_sha256','fold','lambda_rank','sam2_checkpoint_sha256','sam2_config','target_spacing_xyz','image_size','lora','train_ids','validation_ids')
  bad=[k for k in keys if old.get(k)!=meta.get(k)]
  if bad:raise RuntimeError(f'provenance/config mismatch: {bad}')
  if not resume:raise RuntimeError(f'{path.parent} already has a run; use --resume')
 else:
  if resume:raise FileNotFoundError(f'--resume requested but {path} is absent')
  path.write_text(json.dumps(meta,indent=2))
def run(args):
 out=args.out;out.mkdir(parents=True,exist_ok=True);splits=load_splits(SPLIT_PATH);fold=splits['folds'][args.fold];lookup={p.name:p for p in patient_dirs(DATA_ROOT,'train')};train={p:CTVCase(lookup[p]) for p in fold['train']};valid={p:CTVCase(lookup[p]) for p in fold['val']};meta=expected_config(args,fold,train,valid);assert_or_write_config(out/'training_config.json',meta,args.resume)
 device=torch.device(args.device);core,names=build_fullvolume_no_prompt(SAM2_ROOT,MODEL_CFG,CHECKPOINT,device,LORA_RANK,LORA_ALPHA,LORA_DROPOUT);opt=torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=args.lr,weight_decay=WEIGHT_DECAY)
 tr_ids={id(p) for p in core.parameters() if p.requires_grad};op_ids={id(p) for g in opt.param_groups for p in g['params']}
 if tr_ids!=op_ids:raise RuntimeError('optimizer does not cover all trainable parameters')
 start,best,bad=1,-1.,0;last=out/'last.pth'
 if args.resume:
  if not last.exists():raise FileNotFoundError(last)
  state=torch.load(last,map_location='cpu',weights_only=False)
  if state.get('fold')!=args.fold or state.get('lambda_rank')!=args.lambda_rank or state.get('split_sha256')!=meta['split_sha256']:raise RuntimeError('last.pth provenance mismatch')
  core.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng_state']);start=int(state['epoch'])+1;best=float(state['best_validation_dice']);bad=int(state['bad_epochs'])
  if bad>=args.patience or start>args.epochs:
   print(json.dumps({'status':'ALREADY_COMPLETE','epoch':int(state['epoch']),'best_validation_dice':best,'bad_epochs':bad}),flush=True);return best
 else: seed_all(args.seed)
 for epoch in range(start,args.epochs+1):
  core.train();losses=[];segs=[];ranks=[];pairs=[]
  for plan in sample_epoch_plans(train.values(),epoch,args.seed,WINDOW):
   b=collate([train[plan['patient']].video(plan['source_indices'])]).to(device);loss,info=episode(core,b,plan['order'],plan,args.lambda_rank);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()));segs.append(float(info['seg_loss']));ranks.append(float(info['rank_loss']));pairs.append(int(info['pair_count']))
  core.eval();vl=[]
  with torch.no_grad():
   for plan in sample_epoch_plans(valid.values(),0,args.seed,WINDOW):
    b=collate([valid[plan['patient']].video(plan['source_indices'])]).to(device);loss,_=episode(core,b,plan['order'],plan,args.lambda_rank);vl.append(float(loss))
  valrows=full_validate(core,valid,device);mean_dice=float(np.mean([x['dice'] for x in valrows]));mean_hd=float(np.mean([x['hd95_mm'] for x in valrows]));improved=mean_dice>best;bad=0 if improved else bad+1
  state={'schema_version':1,'epoch':epoch,'model':core.state_dict(),'optimizer':opt.state_dict(),'train_loss':float(np.mean(losses)),'validation_loss':float(np.mean(vl)),'validation_dice':mean_dice,'best_validation_dice':max(best,mean_dice),'bad_epochs':bad,'lambda_rank':args.lambda_rank,'fold':args.fold,'split_sha256':meta['split_sha256'],'rng_state':capture_rng()};torch.save(state,last)
  if improved:best=mean_dice;torch.save(state,out/'best.pth');write_rows(out/'best_validation_patient_metrics.csv',valrows)
  row={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_seg_loss':float(np.mean(segs)),'train_rank_loss':float(np.mean(ranks)),'train_transition_pair_count':int(sum(pairs)),'validation_loss':float(np.mean(vl)),'validation_dice_original_space':mean_dice,'validation_hd95_mm_original_space':mean_hd,'best_validation_dice':best,'best_checkpoint_saved':int(improved),'bad_epochs':bad};append_row(out/'epoch_metrics.csv',row);plot_history(out/'epoch_metrics.csv',out);print(json.dumps(row),flush=True)
  if bad>=args.patience:break
 return best
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--lambda-rank',type=float,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=MAX_EPOCHS);p.add_argument('--patience',type=int,default=PATIENCE);p.add_argument('--lr',type=float,default=LR);p.add_argument('--seed',type=int,default=SEED);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.lambda_rank not in LAMBDA_GRID:raise RuntimeError(f'unsupported lambda {a.lambda_rank}')
 run(a)
if __name__=='__main__':main()