from types import MethodType
import torch
import torch.nn.functional as F
def heads(self,backbone_features,point_inputs=None,mask_inputs=None,high_res_features=None,multimask_output=False):
 b=backbone_features.size(0);d=backbone_features.device
 if point_inputs is None:coords=torch.zeros(b,1,2,device=d);labels=-torch.ones(b,1,dtype=torch.int32,device=d)
 else:coords,labels=point_inputs['point_coords'],point_inputs['point_labels']
 prompt=None if mask_inputs is None else (F.interpolate(mask_inputs.float(),size=self.sam_prompt_encoder.mask_input_size,mode='bilinear',align_corners=False,antialias=True) if mask_inputs.shape[-2:]!=self.sam_prompt_encoder.mask_input_size else mask_inputs)
 sparse,dense=self.sam_prompt_encoder(points=(coords,labels),boxes=None,masks=prompt)
 ctx=getattr(self,'_multigtv_context',None)
 if ctx is not None:
  if mask_inputs is None:raise RuntimeError('current GTV mask prompt required')
  previous,center,following=ctx
  if not torch.equal(center.to(mask_inputs.device),mask_inputs):raise RuntimeError('GTV center/context mismatch')
  def enc(m):
   m=F.interpolate(m.float(),size=self.sam_prompt_encoder.mask_input_size,mode='bilinear',align_corners=False,antialias=True) if m.shape[-2:]!=self.sam_prompt_encoder.mask_input_size else m
   return self.sam_prompt_encoder(points=None,boxes=None,masks=m)[1]
  dense=self.gtv_prompt_fusion(enc(previous),dense,enc(following))
 low,ious,tokens,obj=self.sam_mask_decoder(image_embeddings=backbone_features,image_pe=self.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=multimask_output,repeat_image=False,high_res_features=high_res_features)
 appearing=obj>0 if self.pred_obj_scores else None;low=low.float();high=F.interpolate(low,size=(self.image_size,self.image_size),mode='bilinear',align_corners=False);tok=tokens[:,0]
 if multimask_output:
  ix=torch.argmax(ious,-1);bi=torch.arange(b,device=d);lowmask=low[bi,ix].unsqueeze(1);highmask=high[bi,ix].unsqueeze(1);tok=tokens[bi,ix] if tokens.size(1)>1 else tok
 else:lowmask,highmask=low,high
 ptr=self.obj_ptr_proj(tok)
 if self.pred_obj_scores:
  a=obj.sigmoid() if self.soft_no_obj_ptr else appearing.float()
  if self.fixed_no_obj_ptr:ptr=a*ptr
  ptr=ptr+(1-a)*self.no_obj_ptr
 return low,high,ious,lowmask,highmask,ptr,obj
def install(model):
 if not model.pred_obj_scores:raise RuntimeError('MultiGTV requires pred_obj_scores=True')
 from fusion import MultiSliceGTVPromptFusion
 model._forward_sam_heads=MethodType(heads,model);model._no_mask_hard_gate=True;model.gtv_prompt_fusion=MultiSliceGTVPromptFusion().to(next(model.parameters()).device);return model