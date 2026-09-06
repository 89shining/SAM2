from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice(pred,gt):
    pred=np.asarray(pred,bool);gt=np.asarray(gt,bool)
    return float((2*(pred&gt).sum()+1e-6)/(pred.sum()+gt.sum()+1e-6))
def _surface(mask):
    if not mask.any(): return mask
    return mask ^ binary_erosion(mask)
def hd95(pred,gt,spacing_zyx):
    pred=np.asarray(pred,bool);gt=np.asarray(gt,bool)
    if not pred.any() and not gt.any(): return 0.0
    if not pred.any() or not gt.any(): return float('inf')
    ps,gs=_surface(pred),_surface(gt)
    dgt=distance_transform_edt(~gs,sampling=spacing_zyx); dp=distance_transform_edt(~ps,sampling=spacing_zyx)
    return float(np.percentile(np.concatenate((dgt[ps],dp[gs])),95))
def asd(pred,gt,spacing_zyx):
    pred=np.asarray(pred,bool);gt=np.asarray(gt,bool)
    if not pred.any() and not gt.any(): return 0.0
    if not pred.any() or not gt.any(): return float('inf')
    ps,gs=_surface(pred),_surface(gt);dgt=distance_transform_edt(~gs,sampling=spacing_zyx);dp=distance_transform_edt(~ps,sampling=spacing_zyx)
    a=np.concatenate((dgt[ps],dp[gs]));return float(a.mean())
def boundary_metrics(pred,gt,spacing_z):
    p=np.asarray(pred,bool).any(axis=(1,2)); g=np.asarray(gt,bool).any(axis=(1,2)); gz=np.where(g)[0];pz=np.where(p)[0]
    if not len(gz): raise RuntimeError('ground truth CTV empty')
    sup,inf=int(gz[0]),int(gz[-1])
    if len(pz): psup,pinf=int(pz[0]),int(pz[-1])
    else: psup,pinf=-1,-1
    return {'gt_superior_slice':sup,'gt_inferior_slice':inf,'pred_superior_slice':psup,'pred_inferior_slice':pinf,'superior_error_slices':abs(psup-sup) if psup>=0 else float('inf'),'inferior_error_slices':abs(pinf-inf) if pinf>=0 else float('inf'),'superior_mae_mm':abs(psup-sup)*spacing_z if psup>=0 else float('inf'),'inferior_mae_mm':abs(pinf-inf)*spacing_z if pinf>=0 else float('inf'),'outside_fp_superior':int(p[:sup].sum()),'outside_fp_inferior':int(p[inf+1:].sum()),'outside_fp_total':int(p[:sup].sum()+p[inf+1:].sum())}