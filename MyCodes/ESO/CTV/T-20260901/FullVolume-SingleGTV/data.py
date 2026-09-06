from __future__ import annotations
import json,random
from pathlib import Path
import numpy as np,SimpleITK as sitk,torch
import torch.nn.functional as F
from training.utils.data_utils import Frame,Object,VideoDatapoint
from config import TARGET_SPACING_XYZ,IMAGE_SIZE

def patient_dirs(root,partition='train'): return sorted((Path(root)/partition).glob('p_*'),key=lambda p:int(p.name[2:]))
def load_splits(path):
 x=json.loads(Path(path).read_text());f=x.get('folds')
 if not isinstance(f,list) or len(f)!=5: raise RuntimeError('expected 5 folds')
 for i,a in enumerate(f):
  if a.get('fold')!=i or len(a.get('train',[]))!=72 or len(a.get('val',[]))!=18: raise RuntimeError(f'invalid fold {i}')
 return x
def make_reference(image,spacing_xyz=TARGET_SPACING_XYZ):
 size=np.maximum(1,np.round((np.array(image.GetSize())-1)*np.array(image.GetSpacing())/np.array(spacing_xyz)).astype(int)+1);ref=sitk.Image([int(x) for x in size],image.GetPixelID());ref.SetSpacing(spacing_xyz);ref.SetOrigin(image.GetOrigin());ref.SetDirection(image.GetDirection());return ref
def resample(image,ref,mask=False): return sitk.Resample(image,ref,sitk.Transform(),sitk.sitkNearestNeighbor if mask else sitk.sitkLinear,0.,image.GetPixelID())
def _frame(rgb,mask,i):
 x=F.interpolate(torch.from_numpy(rgb).float().unsqueeze(0),size=(IMAGE_SIZE,IMAGE_SIZE),mode='bilinear',align_corners=False).squeeze(0);m=F.interpolate(torch.from_numpy(mask.astype('float32')).unsqueeze(0).unsqueeze(0),size=(IMAGE_SIZE,IMAGE_SIZE),mode='nearest').squeeze().bool();return Frame(data=x,objects=[Object(object_id=1,frame_index=i,segment=m)])
def _prompt(mask): return F.interpolate(torch.from_numpy(mask.astype('float32')).unsqueeze(0).unsqueeze(0),size=(IMAGE_SIZE,IMAGE_SIZE),mode='nearest').squeeze(0).bool()
class CTVGTVCase:
 def __init__(self,pdir):
  self.pdir=Path(pdir);self.patient=self.pdir.name;self.original_ct=sitk.ReadImage(str(self.pdir/'image.nii.gz'));self.original_ctv=sitk.ReadImage(str(self.pdir/'CTV.nii.gz'));self.original_gtv=sitk.ReadImage(str(self.pdir/'GTV.nii.gz'));self.reference=make_reference(self.original_ct);ct=resample(self.original_ct,self.reference);cv=resample(self.original_ctv,self.reference,True);gv=resample(self.original_gtv,self.reference,True);self.image=sitk.GetArrayFromImage(ct).astype('float32');self.ctv=sitk.GetArrayFromImage(cv)>0;self.gtv=sitk.GetArrayFromImage(gv)>0
  if not(self.image.shape==self.ctv.shape==self.gtv.shape):raise RuntimeError(f'{self.patient}: shape mismatch')
  z=np.where(self.ctv.any((1,2)))[0]
  if not len(z):raise RuntimeError(f'{self.patient}: empty CTV')
  self.z_sup,self.z_inf=int(z[0]),int(z[-1]);self.rgb=np.repeat(((np.clip(self.image,-160,240)+160)/400)[:,None],3,1);self.spacing_zyx=tuple(reversed(self.original_ct.GetSpacing()))
 def video(self,targets):
  frames=[_frame(self.rgb[z],self.ctv[z],i) for i,z in enumerate(targets)];prompts=torch.stack([_prompt(self.gtv[z]) for z in targets]);return VideoDatapoint(frames=frames,video_id=int(self.patient[2:]),size=(IMAGE_SIZE,IMAGE_SIZE)),prompts
 def restore_probability_to_original(self,p):
  if p.shape!=self.ctv.shape:raise RuntimeError('probability shape mismatch')
  im=sitk.GetImageFromArray(np.asarray(p,dtype='float32'));im.CopyInformation(self.reference);return sitk.GetArrayFromImage(sitk.Resample(im,self.original_ctv,sitk.Transform(),sitk.sitkLinear,0.,sitk.sitkFloat32)).astype('float32')
def sample_epoch_plans(cases,epoch,seed,length=8):
 rng=random.Random(seed+epoch);out=[]
 for c in cases:
  n=len(c.image);sup,inf=c.z_sup,c.z_inf;valid=list(range(n-length+1));cats=[('internal',[s for s in valid if s>=sup and s+length-1<=inf]),('superior_crossing',[s for s in valid if s<sup<=s+length-1]),('inferior_crossing',[s for s in valid if s<=inf<s+length-1]),('superior_negative',[s for s in valid if s+length-1<sup]),('inferior_negative',[s for s in valid if s>inf])]
  for kind,starts in cats:
   if not starts:raise RuntimeError(f'{c.patient}: no {kind}')
   s=rng.choice(starts);forward=rng.random()<.5;out.append({'patient':c.patient,'category':kind,'source_indices':list(range(s,s+length)),'order':list(range(length)) if forward else list(range(length-1,-1,-1)),'z_sup':sup,'z_inf':inf})
 return out