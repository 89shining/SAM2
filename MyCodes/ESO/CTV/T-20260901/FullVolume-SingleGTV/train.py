from __future__ import annotations
import argparse,csv,hashlib,json,random
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,SimpleITK as sitk
from config import *
from data import CTVGTVCase,patient_dirs,load_splits,sample_epoch_plans
from model import build_singlegtv
from engine import episode,midpoint_bidirectional
from metrics import dice,hd95,asd,boundary_metrics
def collate(v):
 from training.utils.data_utils import collate_fn
 return collate_fn(v,dict_key='fullvolume_singlegtv_ctv')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def setrng(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda']) if torch.cuda.is_available() and x.get('cuda') else None
def write(path,rows,append=False):
 mode='a' if append else 'w';exists=Path(path).exists() and append
 with Path(path).open(mode,newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));(None if exists else w.writeheader());w.writerows(rows)
def plot_curve(path,out):
 try:
  import matplotlib.pyplot as plt
  rows=list(csv.DictReader(Path(path).open()));e=[int(r['epoch']) for r in rows];plt.figure(figsize=(8,5));plt.plot(e,[float(r['train_loss']) for r in rows],label='train loss');plt.plot(e,[float(r['validation_loss']) for r in rows],label='validation loss');plt.plot(e,[float(r['validation_dice_original_space']) for r in rows],label='validation Dice3D');plt.legend();plt.xlabel('epoch');plt.tight_layout();plt.savefig(Path(out)/'training_curve.png',dpi=160);plt.close()
 except Exception as exc:Path(out,'training_curve_warning.txt').write_text(str(exc))
def predprob(case,preds):
 q=torch.cat([preds[i] for i in range(len(preds))]).sigmoid();q=F.interpolate(q,size=case.ctv.shape[-2:],mode='bilinear',align_corners=False);return q[:,0].cpu().numpy().astype('float32')
def validate(core,cases,dev):
 rows=[];core.eval()
 for c in cases.values():
  b,p=c.video(list(range(len(c.image))));b=collate([b]).to(dev);q=predprob(c,midpoint_bidirectional(core,b,p));o=c.restore_probability_to_original(q)>.5;g=sitk.GetArrayFromImage(c.original_ctv)>0;r={'patient':c.patient,'dice':dice(o,g),'hd95_mm':hd95(o,g,c.spacing_zyx),'asd_mm':asd(o,g,c.spacing_zyx)};r.update(boundary_metrics(o,g,c.spacing_zyx[0]));rows.append(r)
 return rows
def materialize(core,c,plan,dev):
 b,p=c.video(plan['source_indices']);b=collate([b]).to(dev)
 with torch.no_grad():episode(core,b,p,plan['order'],plan,0.)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--lambda-rank',type=float,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--patience',type=int,default=10);ap.add_argument('--resume',action='store_true');ap.add_argument('--device',default='cuda');a=ap.parse_args();
 if a.lambda_rank not in LAMBDA_GRID:raise RuntimeError('invalid lambda')
 a.out.mkdir(parents=True,exist_ok=True);sp=load_splits(SPLIT_PATH);f=sp['folds'][a.fold];lookup={p.name:p for p in patient_dirs(DATA_ROOT,'train')};tr={x:CTVGTVCase(lookup[x]) for x in f['train']};va={x:CTVGTVCase(lookup[x]) for x in f['val']};meta={'experiment':'Independent FullVolume-SingleGTV','fold':a.fold,'lambda_rank':a.lambda_rank,'split_sha256':sha(SPLIT_PATH),'train_ids':sorted(tr),'validation_ids':sorted(va),'same_control_as':'FullVolume-NoPrompt','trainable':'image Q/V LoRA; prompt encoder; memory encoder; memory attention; mask decoder'};cfg=a.out/'training_config.json'
 if cfg.exists():
  old=json.loads(cfg.read_text());
  if old!=meta:raise RuntimeError('config mismatch')
  if not a.resume:raise RuntimeError('existing run requires --resume')
 else:
  if a.resume:raise RuntimeError('resume missing config')
  cfg.write_text(json.dumps(meta,indent=2))
 dev=torch.device(a.device);core,names=build_singlegtv(SAM2_ROOT,MODEL_CFG,CHECKPOINT,dev,LORA_RANK,LORA_ALPHA,LORA_DROPOUT);first=sample_epoch_plans(tr.values(),1,SEED,WINDOW)[0];materialize(core,tr[first['patient']],first,dev);opt=torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],lr=LR,weight_decay=WEIGHT_DECAY);ids={id(p) for p in core.parameters() if p.requires_grad};oids={id(p) for g in opt.param_groups for p in g['params']};
 if ids!=oids:raise RuntimeError('optimizer coverage mismatch')
 start,best,bad=1,-1.,0;last=a.out/'last.pth'
 if a.resume:
  s=torch.load(last,map_location='cpu',weights_only=False);core.load_state_dict(s['model'],strict=True);opt.load_state_dict(s['optimizer']);setrng(s['rng']);start=s['epoch']+1;best=s['best'];bad=s['bad'];
  if bad>=a.patience or start>a.epochs:print('ALREADY_COMPLETE');return
 else:seed(SEED)
 for e in range(start,a.epochs+1):
  core.train();tl=[];sl=[];rl=[]
  for plan in sample_epoch_plans(tr.values(),e,SEED,WINDOW):
   b,p=tr[plan['patient']].video(plan['source_indices']);b=collate([b]).to(dev);loss,info=episode(core,b,p,plan['order'],plan,a.lambda_rank);opt.zero_grad(set_to_none=True);loss.backward();opt.step();tl.append(float(loss.detach()));sl.append(float(info['seg_loss']));rl.append(float(info['rank_loss']))
  core.eval();vl=[]
  with torch.no_grad():
   for plan in sample_epoch_plans(va.values(),0,SEED,WINDOW):
    b,p=va[plan['patient']].video(plan['source_indices']);b=collate([b]).to(dev);loss,_=episode(core,b,p,plan['order'],plan,a.lambda_rank);vl.append(float(loss))
  vr=validate(core,va,dev);d=float(np.mean([x['dice'] for x in vr]));h=float(np.mean([x['hd95_mm'] for x in vr]));im=d>best;bad=0 if im else bad+1;s={'epoch':e,'model':core.state_dict(),'optimizer':opt.state_dict(),'best':max(best,d),'bad':bad,'rng':rng()};torch.save(s,last)
  if im:best=d;torch.save(s,a.out/'best.pth');write(a.out/'best_validation_patient_metrics.csv',vr)
  row={'epoch':e,'train_loss':np.mean(tl),'train_seg_loss':np.mean(sl),'train_rank_loss':np.mean(rl),'validation_loss':np.mean(vl),'validation_dice_original_space':d,'validation_hd95_mm_original_space':h,'best_validation_dice':best,'best_checkpoint_saved':int(im),'bad_epochs':bad};write(a.out/'epoch_metrics.csv',[row],append=True);plot_curve(a.out/'epoch_metrics.csv',a.out);print(json.dumps(row),flush=True)
  if bad>=a.patience:break
if __name__=='__main__':main()