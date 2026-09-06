from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from training.utils.data_utils import Frame, Object, VideoDatapoint
from config import TARGET_SPACING_XYZ, IMAGE_SIZE

def patient_dirs(root, partition='train'):
    return sorted((Path(root)/partition).glob('p_*'), key=lambda p:int(p.name[2:]))

def load_splits(path):
    x=json.loads(Path(path).read_text())
    folds=x.get('folds');
    if not isinstance(folds,list) or len(folds)!=5: raise RuntimeError('shared_splits.json must define exactly 5 folds')
    for i,f in enumerate(folds):
        if f.get('fold')!=i or len(f.get('train',[]))!=72 or len(f.get('val',[]))!=18: raise RuntimeError(f'invalid fold {i}')
    return x

def make_reference(image, spacing_xyz=TARGET_SPACING_XYZ):
    old_size=np.array(image.GetSize(),dtype=float); old_spacing=np.array(image.GetSpacing(),dtype=float); new_spacing=np.array(spacing_xyz,dtype=float)
    new_size=np.maximum(1,np.round((old_size-1)*old_spacing/new_spacing).astype(int)+1)
    ref=sitk.Image([int(x) for x in new_size], image.GetPixelID())
    ref.SetSpacing(tuple(new_spacing)); ref.SetOrigin(image.GetOrigin()); ref.SetDirection(image.GetDirection())
    return ref

def resample(image, reference, is_mask=False):
    return sitk.Resample(image, reference, sitk.Transform(), sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear, 0.0, image.GetPixelID())

def _frame(rgb, mask, i):
    image=F.interpolate(torch.from_numpy(rgb).float().unsqueeze(0),size=(IMAGE_SIZE,IMAGE_SIZE),mode='bilinear',align_corners=False).squeeze(0)
    seg=F.interpolate(torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),size=(IMAGE_SIZE,IMAGE_SIZE),mode='nearest').squeeze(0).squeeze(0).bool()
    return Frame(data=image, objects=[Object(object_id=1,frame_index=i,segment=seg)])

class CTVCase:
    def __init__(self, pdir):
        self.pdir=Path(pdir); self.patient=self.pdir.name
        self.original_ct=sitk.ReadImage(str(self.pdir/'image.nii.gz')); self.original_ctv=sitk.ReadImage(str(self.pdir/'CTV.nii.gz'))
        self.reference=make_reference(self.original_ct)
        ct=resample(self.original_ct,self.reference,False); ctv=resample(self.original_ctv,self.reference,True)
        self.image=sitk.GetArrayFromImage(ct).astype(np.float32); self.ctv=(sitk.GetArrayFromImage(ctv)>0)
        if self.image.shape!=self.ctv.shape: raise RuntimeError(f'{self.patient}: CT/CTV shape mismatch')
        x=np.clip(self.image,-160.,240.); self.rgb=np.repeat(((x+160.)/400.)[:,None],3,axis=1)
        z=np.where(self.ctv.any(axis=(1,2)))[0]
        if not len(z): raise RuntimeError(f'{self.patient}: empty CTV')
        self.z_sup,self.z_inf=int(z[0]),int(z[-1]); self.spacing_zyx=tuple(reversed(self.original_ct.GetSpacing()))
    def video(self, indices):
        return VideoDatapoint(frames=[_frame(self.rgb[z],self.ctv[z],i) for i,z in enumerate(indices)], video_id=int(self.patient[2:]), size=(IMAGE_SIZE,IMAGE_SIZE))
    def restore_probability_to_original(self, probability_zyx):
        """Resample continuous probability to original CTV geometry before thresholding."""
        if probability_zyx.shape != self.ctv.shape: raise RuntimeError('probability must be in resampled geometry')
        im=sitk.GetImageFromArray(np.asarray(probability_zyx,dtype=np.float32)); im.CopyInformation(self.reference)
        original=sitk.Resample(im,self.original_ctv,sitk.Transform(),sitk.sitkLinear,0.0,sitk.sitkFloat32)
        return sitk.GetArrayFromImage(original).astype(np.float32)

def _choose(rng, values, fallback): return rng.choice(values) if values else fallback

def sample_epoch_plans(cases, epoch, seed, length=8):
    rng=random.Random(seed+epoch); plans=[]
    for c in cases:
        n=len(c.image); sup,inf=c.z_sup,c.z_inf
        valid=list(range(0,n-length+1)); internal=[s for s in valid if s>=sup and s+length-1<=inf]
        sup_cross=[s for s in valid if s<sup<=s+length-1]
        inf_cross=[s for s in valid if s<=inf<s+length-1]
        sup_neg=[s for s in valid if s+length-1<sup]; inf_neg=[s for s in valid if s>inf]
        cats=[('internal',internal),('superior_crossing',sup_cross),('inferior_crossing',inf_cross),('superior_negative',sup_neg),('inferior_negative',inf_neg)]
        for category,starts in cats:
            if not starts: raise RuntimeError(f'{c.patient}: no legal {category} window')
            start=_choose(rng,starts,starts[0]); forward=rng.random()<0.5
            plans.append({'patient':c.patient,'category':category,'source_indices':list(range(start,start+length)),'order':list(range(length)) if forward else list(range(length-1,-1,-1)),'direction':'forward' if forward else 'reverse','z_sup':sup,'z_inf':inf})
    return plans