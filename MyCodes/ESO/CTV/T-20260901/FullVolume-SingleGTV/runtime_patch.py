from types import MethodType
import torch
import torch.nn.functional as F

def heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
    b, d = backbone_features.size(0), backbone_features.device
    if point_inputs is None:
        coords = torch.zeros(b, 1, 2, device=d)
        labels = -torch.ones(b, 1, dtype=torch.int32, device=d)
    else:
        coords, labels = point_inputs['point_coords'], point_inputs['point_labels']
    prompt = mask_inputs
    if prompt is not None and prompt.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
        prompt = F.interpolate(prompt.float(), size=self.sam_prompt_encoder.mask_input_size, mode='bilinear', align_corners=False, antialias=True)
    sparse, dense = self.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=prompt)
    low, ious, tokens, obj = self.sam_mask_decoder(
        image_embeddings=backbone_features, image_pe=self.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=multimask_output, repeat_image=False, high_res_features=high_res_features,
    )
    appearing = obj > 0 if self.pred_obj_scores else None
    low = low.float()
    high = F.interpolate(low, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
    tok = tokens[:, 0]
    if multimask_output:
        ix = torch.argmax(ious, -1); bi = torch.arange(b, device=d)
        lowmask, highmask = low[bi, ix].unsqueeze(1), high[bi, ix].unsqueeze(1)
        tok = tokens[bi, ix] if tokens.size(1) > 1 else tok
    else:
        lowmask, highmask = low, high
    ptr = self.obj_ptr_proj(tok)
    if self.pred_obj_scores:
        a = obj.sigmoid() if self.soft_no_obj_ptr else appearing.float()
        if self.fixed_no_obj_ptr: ptr = a * ptr
        ptr = ptr + (1 - a) * self.no_obj_ptr
    return low, high, ious, lowmask, highmask, ptr, obj

def install(model):
    if not model.pred_obj_scores:
        raise RuntimeError('SingleGTV requires pred_obj_scores=True')
    model._forward_sam_heads = MethodType(heads, model)
    model._no_mask_hard_gate = True
    return model