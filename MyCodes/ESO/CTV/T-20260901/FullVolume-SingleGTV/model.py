from __future__ import annotations
import sys, torch
from pathlib import Path
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

def build_singlegtv(sam2_root, model_cfg, checkpoint, device, lora_rank, lora_alpha, lora_dropout):
    root = str(Path(sam2_root))
    if root not in sys.path: sys.path.insert(0, root)
    from training.model.sam2 import SAM2Train
    from runtime_patch import install
    if GlobalHydra.instance().is_initialized(): GlobalHydra.instance().clear()
    with initialize_config_module('sam2', version_base='1.2'):
        cfg = compose(config_name=model_cfg)
    d = OmegaConf.to_container(cfg.model, resolve=True)
    d['image_size'] = 1024; d['use_mask_input_as_output_without_sam'] = False; d['pred_obj_scores'] = True
    ie, ma, me = d.pop('image_encoder'), d.pop('memory_attention'), d.pop('memory_encoder'); d.pop('_target_', None)
    m = SAM2Train(image_encoder=instantiate(ie, _recursive_=True), memory_attention=instantiate(ma, _recursive_=True), memory_encoder=instantiate(me, _recursive_=True), prob_to_use_pt_input_for_train=0., prob_to_use_pt_input_for_eval=0., prob_to_use_box_input_for_train=0., prob_to_use_box_input_for_eval=0., prob_to_sample_from_gt_for_train=0., num_frames_to_correct_for_train=0, num_frames_to_correct_for_eval=0, rand_frames_to_correct_for_train=False, rand_init_cond_frames_for_train=False, rand_init_cond_frames_for_eval=False, add_all_frames_to_correct_as_cond=False, num_correction_pt_per_frame=0, **d)
    install(m)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False); m.load_state_dict(state.get('model', state), strict=False)
    for p in m.parameters(): p.requires_grad = False
    from sam2.modeling.lora import LoRAConfig, apply_qv_lora_to_fused_qkv
    count = apply_qv_lora_to_fused_qkv(m, LoRAConfig(r=lora_rank, alpha=int(lora_alpha), dropout=lora_dropout, target_modules=('qkv',), target_prefixes=('image_encoder',), freeze_base_model=False))
    for n,p in m.named_parameters():
        if (n.startswith('image_encoder.') and 'lora_' in n) or n.startswith('sam_prompt_encoder.') or n.startswith('memory_encoder.') or n.startswith('memory_attention.') or n.startswith('sam_mask_decoder.'):
            p.requires_grad = True
    names = [n for n,p in m.named_parameters() if p.requires_grad]
    if count <= 0 or not any(n.startswith('sam_prompt_encoder.') for n in names) or m.use_mask_input_as_output_without_sam:
        raise RuntimeError('invalid SingleGTV model setup')
    return m.to(device), names