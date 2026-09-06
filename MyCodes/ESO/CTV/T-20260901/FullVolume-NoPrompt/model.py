from __future__ import annotations
import sys, torch
from pathlib import Path
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _instantiate(sam2_root, model_cfg):
    root=str(Path(sam2_root));
    if root not in sys.path: sys.path.insert(0, root)
    from training.model.sam2 import SAM2Train
    if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
    with initialize_config_module('sam2', version_base='1.2'):
        cfg=compose(config_name=model_cfg)
    d=OmegaConf.to_container(cfg.model, resolve=True)
    d['image_size']=1024; d['pred_obj_scores']=False; d['fixed_no_obj_ptr']=False
    ie=d.pop('image_encoder'); ma=d.pop('memory_attention'); me=d.pop('memory_encoder'); d.pop('_target_',None)
    return SAM2Train(image_encoder=instantiate(ie,_recursive_=True), memory_attention=instantiate(ma,_recursive_=True), memory_encoder=instantiate(me,_recursive_=True), prob_to_use_pt_input_for_train=0., prob_to_use_pt_input_for_eval=0., prob_to_use_box_input_for_train=0., prob_to_use_box_input_for_eval=0., prob_to_sample_from_gt_for_train=0., num_frames_to_correct_for_train=0, num_frames_to_correct_for_eval=0, rand_frames_to_correct_for_train=False, rand_frames_to_correct_for_eval=False, add_all_frames_to_correct_as_cond=False, num_correction_pt_per_frame=0, rand_init_cond_frames_for_train=False, rand_init_cond_frames_for_eval=False, **d)

def build_fullvolume_no_prompt(sam2_root, model_cfg, pretrained, device, lora_rank, lora_alpha, lora_dropout=0.0, checkpoint=None):
    model=_instantiate(sam2_root, model_cfg)
    state=torch.load(pretrained, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model',state), strict=False)
    for p in model.parameters(): p.requires_grad=False
    from sam2.modeling.lora import LoRAConfig, apply_qv_lora_to_fused_qkv
    lora_count=apply_qv_lora_to_fused_qkv(model, LoRAConfig(r=lora_rank, alpha=int(lora_alpha), dropout=lora_dropout, target_modules=('qkv',), target_prefixes=('image_encoder',), freeze_base_model=False))
    for n,p in model.named_parameters():
        if n.startswith('image_encoder.') and ('lora_A' in n or 'lora_B' in n): p.requires_grad=True
        elif n.startswith('memory_encoder.') or n.startswith('memory_attention.') or n.startswith('sam_mask_decoder.'): p.requires_grad=True
    if checkpoint is not None:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False); model.load_state_dict(ck['model'],strict=True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if lora_count <= 0 or not any('lora_A' in n for n in names): raise RuntimeError('Image-encoder Q/V LoRA injection failed')
    forbidden=[n for n in names if n.startswith('prompt_encoder.') or (n.startswith('image_encoder.') and 'lora_' not in n)]
    if forbidden: raise RuntimeError(f'Invalid FullVolume-NoPrompt trainable profile: {forbidden[:3]}')
    return model.to(device), names