from __future__ import annotations
import torch
import torch.nn.functional as F

def chunk(core, x, size=8):
    if x.shape[0] <= size: return core.forward_image(x)
    cs = [core.forward_image(x[i:i+size]) for i in range(0, x.shape[0], size)]; out = {}
    for k,v in cs[0].items(): out[k] = [torch.cat([c[k][j] for c in cs], 0) for j in range(len(v))] if isinstance(v,(list,tuple)) else torch.cat([c[k] for c in cs],0)
    return out

def segloss(logits, target):
    t = F.interpolate(target.float(), size=logits.shape[-2:], mode='nearest'); b = F.binary_cross_entropy_with_logits(logits,t)
    if bool(t.any()):
        p = torch.sigmoid(logits); return b + 1 - (2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)
    return b

def rankloss(scores, sup, inf, side):
    pairs=[]
    for d in (1,2,3):
        i,o = (sup+d-1,sup-d) if side=='superior' else (inf-d+1,inf+d)
        if i in scores and o in scores: pairs.append((i,o))
    if not pairs: raise RuntimeError('missing transition pairs')
    return torch.stack([F.softplus(scores[o]-scores[i]) for i,o in pairs]).mean(), len(pairs)

def episode(core, batch, prompts, order, plan=None, lam=0.):
    base=chunk(core,batch.flat_img_batch); _,vision,pos,sizes=core._prepare_backbone_features(base)
    outputs={'cond_frame_outputs':{},'non_cond_frame_outputs':{}}; losses=[]; scores={}; preds={}; pairs=0
    for step,frame in enumerate(order):
        ids=batch.flat_obj_to_img_idx[frame]
        current=[x[:,ids] for x in vision]; vp=[x[:,ids] for x in pos]
        mask=prompts[frame].unsqueeze(0).to(current[0].device)
        out=core.track_step(frame_idx=frame,is_init_cond_frame=(step==0),current_vision_feats=current,current_vision_pos_embeds=vp,feat_sizes=sizes,point_inputs=None,mask_inputs=mask,gt_masks=None,frames_to_add_correction_pt=[],output_dict=outputs,num_frames=int(batch.num_frames),track_in_reverse=order[0]>order[-1],prev_sam_mask_logits=None)
        (outputs['cond_frame_outputs'] if step==0 else outputs['non_cond_frame_outputs'])[frame]=out
        losses.append(segloss(out['pred_masks'],batch.masks[frame].unsqueeze(1).to(out['pred_masks'].device))); preds[frame]=out['pred_masks_high_res']
        if plan is not None:
            mass=torch.sigmoid(out['pred_masks']).mean((-2,-1)).clamp(1e-6,1-1e-6); scores[plan['source_indices'][frame]]=torch.logit(mass).mean()
    if len(losses)!=len(order): raise RuntimeError('all targets must supervise')
    seg=torch.stack(losses).mean(); rank=seg.new_zeros(())
    if plan is not None and lam>0 and plan['category'] in ('superior_crossing','inferior_crossing'):
        rank,pairs=rankloss(scores,plan['z_sup'],plan['z_inf'],'superior' if plan['category']=='superior_crossing' else 'inferior')
    return (1-lam)*seg+lam*rank,{'seg_loss':seg.detach(),'rank_loss':rank.detach(),'pair_count':pairs,'preds':preds}

@torch.no_grad()
def midpoint_bidirectional(core,batch,prompts):
    n=int(batch.num_frames); mid=(n-1)//2; out={}
    for order in (list(range(mid,n)),list(range(mid,-1,-1))): out.update(episode(core,batch,prompts,order)[1]['preds'])
    if len(out)!=n: raise RuntimeError('incomplete inference')
    return out