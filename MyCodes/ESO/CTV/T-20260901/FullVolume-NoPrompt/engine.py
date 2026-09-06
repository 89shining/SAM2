from __future__ import annotations
import torch
import torch.nn.functional as F

def _empty(n,d): return {'point_coords':torch.zeros((n,0,2),device=d),'point_labels':torch.zeros((n,0),dtype=torch.int32,device=d)}
def _chunked_image(core, images, size=8):
    if images.shape[0] <= size: return core.forward_image(images)
    chunks=[core.forward_image(images[i:i+size]) for i in range(0,images.shape[0],size)]
    merged={}
    for key,value in chunks[0].items():
        merged[key]=[torch.cat([chunk[key][j] for chunk in chunks],0) for j in range(len(value))] if isinstance(value,(list,tuple)) else torch.cat([chunk[key] for chunk in chunks],0)
    return merged

def segmentation_loss(logits,target):
    target=F.interpolate(target.float(),size=logits.shape[-2:],mode='nearest')
    bce=F.binary_cross_entropy_with_logits(logits,target)
    if bool(target.any()):
        p=torch.sigmoid(logits); dice=1-(2*(p*target).sum()+1e-6)/(p.sum()+target.sum()+1e-6)
        return dice+bce, {'dice':float(dice.detach()),'bce':float(bce.detach()),'positive':1}
    return bce, {'dice':None,'bce':float(bce.detach()),'positive':0}

def transition_ranking_loss(score_by_physical_z,z_sup,z_inf,side):
    pairs=[]
    for d in (1,2,3):
        inside,outside=(z_sup+d-1,z_sup-d) if side=='superior' else (z_inf-d+1,z_inf+d)
        if inside in score_by_physical_z and outside in score_by_physical_z:pairs.append((inside,outside))
    if not pairs: raise RuntimeError(f'{side} crossing episode has no transition pair')
    values=[F.softplus(score_by_physical_z[o]-score_by_physical_z[i]) for i,o in pairs]
    return torch.stack(values).mean(),len(pairs)

def episode(core,batch,order,plan=None,lambda_rank=0.0):
    base=_chunked_image(core,batch.flat_img_batch,8); _,vision_feats,vision_pos,feat_sizes=core._prepare_backbone_features(base)
    output_dict={'cond_frame_outputs':{},'non_cond_frame_outputs':{}}; prev=None; losses=[];preds={};stats=[];scores={}
    for j,frame in enumerate(order):
        ids=batch.flat_obj_to_img_idx[frame]; vf=[x[:,ids] for x in vision_feats];vp=[x[:,ids] for x in vision_pos]
        out=core.track_step(frame_idx=frame,is_init_cond_frame=(j==0),current_vision_feats=vf,current_vision_pos_embeds=vp,feat_sizes=feat_sizes,point_inputs=None if j==0 else _empty(prev.shape[0],prev.device),mask_inputs=None,gt_masks=None,frames_to_add_correction_pt=[],output_dict=output_dict,num_frames=int(batch.num_frames),track_in_reverse=order[0]>order[-1],prev_sam_mask_logits=prev)
        (output_dict['cond_frame_outputs'] if j==0 else output_dict['non_cond_frame_outputs'])[frame]=out
        loss,st=segmentation_loss(out['pred_masks'],batch.masks[frame].unsqueeze(1).float().to(out['pred_masks'].device));losses.append(loss);stats.append(st);preds[frame]=out['pred_masks_high_res'];prev=out['pred_masks'].detach().clamp(-32,32)
        if plan is not None:
            mass=torch.sigmoid(out['pred_masks']).mean(dim=(-2,-1)).clamp(1e-6,1-1e-6); scores[plan['source_indices'][frame]]=torch.logit(mass).mean()
    if len(losses)!=8:raise RuntimeError('all 8 target slices must be supervised')
    seg=torch.stack(losses).mean(); rank=seg.new_zeros(());pairs=0
    if plan is not None and lambda_rank>0 and plan['category'] in ('superior_crossing','inferior_crossing'):
        side='superior' if plan['category']=='superior_crossing' else 'inferior';rank,pairs=transition_ranking_loss(scores,plan['z_sup'],plan['z_inf'],side)
    total=(1-lambda_rank)*seg+lambda_rank*rank
    return total, {'seg_loss':seg.detach(),'rank_loss':rank.detach(),'pair_count':pairs,'stats':stats,'preds':preds}

@torch.no_grad()
def inference_episode(core,batch,order):
    base=_chunked_image(core,batch.flat_img_batch,8);_,vision_feats,vision_pos,feat_sizes=core._prepare_backbone_features(base)
    output_dict={'cond_frame_outputs':{},'non_cond_frame_outputs':{}};prev=None;preds={}
    for j,frame in enumerate(order):
        ids=batch.flat_obj_to_img_idx[frame];vf=[x[:,ids] for x in vision_feats];vp=[x[:,ids] for x in vision_pos]
        out=core.track_step(frame_idx=frame,is_init_cond_frame=(j==0),current_vision_feats=vf,current_vision_pos_embeds=vp,feat_sizes=feat_sizes,point_inputs=None if j==0 else _empty(prev.shape[0],prev.device),mask_inputs=None,gt_masks=None,frames_to_add_correction_pt=[],output_dict=output_dict,num_frames=int(batch.num_frames),track_in_reverse=order[0]>order[-1],prev_sam_mask_logits=prev)
        (output_dict['cond_frame_outputs'] if j==0 else output_dict['non_cond_frame_outputs'])[frame]=out;preds[frame]=out['pred_masks_high_res'];prev=out['pred_masks'].detach().clamp(-32,32)
    return preds
@torch.no_grad()
def midpoint_bidirectional(core,batch):
    n=int(batch.num_frames);mid=(n-1)//2;out={};out.update(inference_episode(core,batch,list(range(mid,n))));out.update(inference_episode(core,batch,list(range(mid,-1,-1))))
    if len(out)!=n:raise RuntimeError('incomplete full-volume inference')
    return out