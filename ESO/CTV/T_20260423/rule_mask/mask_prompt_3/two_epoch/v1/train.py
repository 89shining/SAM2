#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2 finetuning with iterative two-stage prompting (rule middle):
  pass-1: upper + lower prompts
  middle slice: rule-based (same source as rule_mask/mask_prompt_3/one_epoch)
  pass-2: continue from pass-1 memory and inject middle prompt

Example:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_upper_lower_online_hd95_middle_cv.py
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.distributed as dist
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from medpy.metric.binary import hd95 as medpy_hd95

# ---- robust import path setup ----
CURRENT_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    """
    Find SAM2 repo root dynamically to avoid hard-coded parent depth.
    Root must contain both `training` and `sam2` directories.
    """
    for p in [start] + list(start.parents):
        if (p / "training").is_dir() and (p / "sam2").is_dir():
            return p
    env_root = os.environ.get("SAM2_PROJECT_ROOT", "").strip()
    if env_root:
        p = Path(env_root).resolve()
        if (p / "training").is_dir() and (p / "sam2").is_dir():
            return p
    raise RuntimeError(
        "Cannot locate SAM2 project root from script path. "
        "Set SAM2_PROJECT_ROOT to a directory containing 'training' and 'sam2'."
    )


PROJECT_ROOT = _find_project_root(CURRENT_DIR)
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model.sam2 import SAM2Train
from training.utils.data_utils import Frame, Object, VideoDatapoint, collate_fn


# ================= Defaults =================
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_to_uint8(img2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def _sample_aug_params(enable_augment: bool):
    if not enable_augment:
        return None
    return {
        "hflip": random.random() < 0.5,
        "vflip": random.random() < 0.3,
        "rot_k": random.randint(0, 3),
        "contrast": random.uniform(0.9, 1.1),
        "brightness": random.uniform(-0.05, 0.05),
        "noise_std": random.uniform(0.0, 0.02),
    }


def _apply_aug_to_slice(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    aug_params: dict,
):
    if aug_params is None:
        return image_tensor, mask_tensor

    if aug_params["hflip"]:
        image_tensor = torch.flip(image_tensor, dims=[2])
        mask_tensor = torch.flip(mask_tensor, dims=[1])
    if aug_params["vflip"]:
        image_tensor = torch.flip(image_tensor, dims=[1])
        mask_tensor = torch.flip(mask_tensor, dims=[0])
    if aug_params["rot_k"] > 0:
        image_tensor = torch.rot90(image_tensor, aug_params["rot_k"], dims=[1, 2])
        mask_tensor = torch.rot90(mask_tensor, aug_params["rot_k"], dims=[0, 1])

    image_tensor = image_tensor * aug_params["contrast"] + aug_params["brightness"]
    if aug_params["noise_std"] > 0:
        image_tensor = image_tensor + torch.randn_like(image_tensor) * aug_params["noise_std"]
    image_tensor = image_tensor.clamp(0.0, 1.0)
    return image_tensor, mask_tensor


def patient_sort_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def patient_id_from_folder(pdir: Path) -> str:
    m = re.search(r"(\d+)", pdir.name)
    if m is None:
        raise ValueError(f"Cannot parse patient id from folder name: {pdir.name}")
    return f"CTV_{int(m.group(1)):03d}"


def patient_video_num_from_id(patient_id: str) -> int:
    m = re.search(r"(\d+)$", str(patient_id))
    if m is None:
        raise ValueError(f"Cannot parse numeric id from Patient_ID: {patient_id}")
    return int(m.group(1))


class SAM2TrainUpperLowerDynamicMiddleMask(SAM2Train):
    """
    Two-pass training/eval:
    - pass-1: boundary-only (upper + lower mask prompts)
    - select middle frame online from pass-1 prediction
    - pass-2: boundary + selected middle mask prompt
    """

    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                prob_to_use_pt_input_for_train=0.0,
                prob_to_use_pt_input_for_eval=0.0,
                prob_to_use_box_input_for_train=0.0,
                prob_to_use_box_input_for_eval=0.0,
                prob_to_sample_from_gt_for_train=0.0,
                num_frames_to_correct_for_train=1,
                num_frames_to_correct_for_eval=1,
                rand_frames_to_correct_for_train=False,
                rand_frames_to_correct_for_eval=False,
                add_all_frames_to_correct_as_cond=False,
                num_correction_pt_per_frame=0,
                rand_init_cond_frames_for_train=False,
                rand_init_cond_frames_for_eval=False,
            )
        )
        super().__init__(*args, **kwargs)
        self.runtime_middle_prompt_by_video_id = {}
        self.enable_middle_prompt = False

    def set_runtime_middle_prompt_map(self, mapping):
        self.runtime_middle_prompt_by_video_id = {
            int(k): int(v) for k, v in (mapping or {}).items()
        }
        self.enable_middle_prompt = len(self.runtime_middle_prompt_by_video_id) > 0

    def clear_runtime_middle_prompt_map(self):
        self.runtime_middle_prompt_by_video_id = {}
        self.enable_middle_prompt = False

    @staticmethod
    def _choose_fallback_middle(pos_t: torch.Tensor, lower: int, upper: int) -> int:
        middle_candidates = [int(z) for z in pos_t.tolist() if lower < int(z) < upper]
        if len(middle_candidates) == 0:
            return lower
        middle_candidates = sorted(middle_candidates)
        return int(middle_candidates[len(middle_candidates) // 2])

    @staticmethod
    def _valid_mid_for_object(gt_obj_t_hw: torch.Tensor, mid: int, lower: int, upper: int) -> int:
        t_dim = gt_obj_t_hw.shape[0]
        mid = max(0, min(int(mid), t_dim - 1))
        if bool(gt_obj_t_hw[mid].any()):
            return mid
        if bool(gt_obj_t_hw[lower].any()):
            return int(lower)
        if bool(gt_obj_t_hw[upper].any()):
            return int(upper)
        return int(mid)

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        gt_masks_per_frame = {
            frame_idx: masks.unsqueeze(1)
            for frame_idx, masks in enumerate(input.masks)
        }
        num_frames = input.num_frames

        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["num_frames"] = num_frames
        backbone_out["use_pt_input"] = False
        backbone_out["point_inputs_per_frame"] = {}
        backbone_out["frames_to_add_correction_pt"] = []

        masks_tohw = input.masks  # [T, O, H, W]
        if masks_tohw.ndim != 4:
            raise ValueError(f"Expected input.masks to be [T, O, H, W], got {masks_tohw.shape}")

        t_dim, o_dim = masks_tohw.shape[:2]
        if t_dim != num_frames:
            raise ValueError(f"num_frames mismatch: {num_frames} vs {t_dim}")

        obj_video_ids = input.metadata.unique_objects_identifier[0, :, 0].to(torch.long)

        lower_ids = []
        upper_ids = []
        middle_ids = []

        for obj_idx in range(o_dim):
            per_t_has_fg = masks_tohw[:, obj_idx].flatten(1).any(dim=1)
            pos_t = torch.nonzero(per_t_has_fg, as_tuple=False).flatten()

            if pos_t.numel() == 0:
                lower = int(start_frame_idx)
                upper = int(start_frame_idx)
                middle = int(start_frame_idx)
            else:
                lower = int(pos_t.min().item())
                upper = int(pos_t.max().item())

                if self.enable_middle_prompt:
                    video_id = int(obj_video_ids[obj_idx].item())
                    external_mid = self.runtime_middle_prompt_by_video_id.get(video_id)
                    if external_mid is None:
                        external_mid = self._choose_fallback_middle(pos_t, lower, upper)
                    middle = self._valid_mid_for_object(
                        gt_obj_t_hw=masks_tohw[:, obj_idx],
                        mid=int(external_mid),
                        lower=lower,
                        upper=upper,
                    )
                else:
                    middle = None

            lower_ids.append(lower)
            upper_ids.append(upper)
            middle_ids.append(middle)

        init_cond_frames = set(lower_ids + upper_ids)

        if self.enable_middle_prompt:
            middle_valid = [m for m in middle_ids if m is not None]
            init_cond_frames = init_cond_frames.union(set(middle_valid))

        init_cond_frames = sorted(init_cond_frames)

        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames)
            if t not in init_cond_frames
        ]

        backbone_out["mask_inputs_per_frame"] = {}
        for t in init_cond_frames:
            gt_t = gt_masks_per_frame[t]  # [O,1,H,W]
            prompt_t = torch.zeros_like(gt_t)
            for o in range(o_dim):
                if lower_ids[o] == t or upper_ids[o] == t or (
                        self.enable_middle_prompt and middle_ids[o] is not None and middle_ids[o] == t
                ):
                    prompt_t[o] = gt_t[o]
            backbone_out["mask_inputs_per_frame"][t] = prompt_t

        return backbone_out


class UpperLowerVolumeDataset(Dataset):
    """
    Build full-volume videos from 3D NIfTI:
    - each item is one patient volume (no clip split)
    - one object (CTV) per frame
    """

    def __init__(
        self,
        patient_dirs,
        image_name="image.nii.gz",
        mask_name="CTV.nii.gz",
        window_center=40.0,
        window_width=400.0,
        input_size=1024,
        enable_augment=False,
    ):
        self.patient_dirs = list(patient_dirs)
        self.image_name = image_name
        self.mask_name = mask_name
        self.window_center = float(window_center)
        self.window_width = float(window_width)
        self.input_size = int(input_size)
        self.enable_augment = bool(enable_augment)

        self.samples = []

        for pdir in self.patient_dirs:
            img_path = pdir / self.image_name
            gt_path = pdir / self.mask_name
            if not img_path.exists() or not gt_path.exists():
                continue

            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
            gt = (gt > 0).astype(np.uint8)
            pos = np.where(gt.sum(axis=(1, 2)) > 0)[0]
            if len(pos) == 0:
                continue

            patient_id = patient_id_from_folder(pdir)
            video_id = patient_video_num_from_id(patient_id)

            self.samples.append(
                {
                    "pdir": pdir,
                    "patient_id": patient_id,
                    "video_id": video_id,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdir = sample["pdir"]
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.image_name)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.mask_name)))
        gt = (gt > 0).astype(np.uint8)

        frames = []
        aug_params = _sample_aug_params(self.enable_augment)
        h0, w0 = img.shape[1], img.shape[2]
        for local_t in range(img.shape[0]):
            u8 = window_to_uint8(img[local_t], self.window_center, self.window_width)
            rgb = np.stack([u8, u8, u8], axis=0)  # [3, H, W]
            image_tensor = torch.from_numpy(rgb).float() / 255.0
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_tensor = torch.from_numpy(gt[local_t]).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(
                mask_tensor,
                size=(self.input_size, self.input_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.bool)
            image_tensor, mask_tensor = _apply_aug_to_slice(
                image_tensor=image_tensor,
                mask_tensor=mask_tensor,
                aug_params=aug_params,
            )

            frames.append(
                Frame(
                    data=image_tensor,
                    objects=[Object(object_id=1, frame_index=local_t, segment=mask_tensor)],
                )
            )

        return VideoDatapoint(
            frames=frames,
            video_id=int(sample["video_id"]),
            size=(h0, w0),
        )


def load_model_cfg_dict_once(model_cfg: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=model_cfg)
    return OmegaConf.to_container(cfg.model, resolve=True)


def build_model(
    model_cfg_dict_template,
    pretrained_ckpt: Path,
    freeze_image_encoder: bool,
    device: torch.device,
):
    model_cfg_dict = dict(model_cfg_dict_template)
    model_cfg_dict["freeze_image_encoder"] = freeze_image_encoder

    image_encoder_cfg = model_cfg_dict.pop("image_encoder")
    memory_attention_cfg = model_cfg_dict.pop("memory_attention")
    memory_encoder_cfg = model_cfg_dict.pop("memory_encoder")
    model_cfg_dict.pop("_target_", None)

    image_encoder = instantiate(image_encoder_cfg, _recursive_=True)
    memory_attention = instantiate(memory_attention_cfg, _recursive_=True)
    memory_encoder = instantiate(memory_encoder_cfg, _recursive_=True)

    model = SAM2TrainUpperLowerDynamicMiddleMask(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        **model_cfg_dict,
    )

    state = torch.load(str(pretrained_ckpt), map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys while loading pretrained: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[WARN] missing keys while loading pretrained: {len(missing)}")

    model = model.to(device)
    return model


def build_optimizer(model, base_lr: float, vision_lr: float, weight_decay: float):
    image_encoder_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("image_encoder"):
            image_encoder_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if len(other_params) > 0:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if len(image_encoder_params) > 0:
        groups.append({"params": image_encoder_params, "lr": vision_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(groups)


def _dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_dice_ce_loss(outputs, batch_masks: torch.Tensor):
    # Final loss is normalized to keep strict < 1.
    t_dim = batch_masks.shape[0]
    losses = []
    for t in range(t_dim):
        logits = outputs[t]["pred_masks_high_res"][:, 0]  # [O,H,W]
        target = batch_masks[t].float()
        ce_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        ce_loss = ce_raw / (1.0 + ce_raw)  # in (0,1)
        dice_loss = _dice_loss_from_logits(logits, target)
        losses.append(0.5 * (dice_loss + ce_loss))
    if len(losses) == 0:
        return torch.zeros((), device=batch_masks.device, dtype=torch.float32)
    return torch.stack(losses, dim=0).mean()


def compute_batch_volume_dice(outputs, batch_masks: torch.Tensor) -> float:
    # outputs: list[T] each has pred_masks_high_res [O,1,H,W], batch_masks: [T,O,H,W]
    t_dim, o_dim = batch_masks.shape[:2]
    dices = []
    for o in range(o_dim):
        pred_stack = []
        gt_stack = []
        for t in range(t_dim):
            pred = outputs[t]["pred_masks_high_res"][o, 0]
            gt = batch_masks[t, o].float()
            pred_stack.append((pred > 0).float())
            gt_stack.append(gt)
        pred_vol = torch.stack(pred_stack, dim=0)
        gt_vol = torch.stack(gt_stack, dim=0)
        inter = (pred_vol * gt_vol).sum()
        denom = pred_vol.sum() + gt_vol.sum()
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        dices.append(float(dice.item()))
    return float(np.mean(dices)) if len(dices) else 0.0


def _safe_hd95_2d(pred2d: np.ndarray, gt2d: np.ndarray) -> float:
    pred2d = pred2d.astype(bool)
    gt2d = gt2d.astype(bool)

    # 跳过无效slice
    if pred2d.sum() == 0 or gt2d.sum() == 0:
        return -1.0

    try:
        return float(medpy_hd95(pred2d, gt2d, voxelspacing=(1.0, 1.0)))
    except Exception:
        return -1.0


def select_rule_middle_from_gt(batch_masks: torch.Tensor, unique_objects_identifier):
    t_dim, o_dim = batch_masks.shape[:2]
    runtime_map = {}
    gt_vols = batch_masks.detach().cpu().numpy().astype(np.uint8)

    for o in range(o_dim):
        video_id = int(unique_objects_identifier[0, o, 0].item())
        gt_o = gt_vols[:, o]
        pos = np.where(gt_o.reshape(t_dim, -1).any(axis=1))[0]
        if len(pos) == 0:
            runtime_map[video_id] = 0
            continue

        lower = int(pos.min())
        upper = int(pos.max())
        middle_candidates = [int(z) for z in pos.tolist() if lower < int(z) < upper]
        if len(middle_candidates) == 0:
            runtime_map[video_id] = lower
            continue
        middle_candidates = sorted(middle_candidates)
        runtime_map[video_id] = int(middle_candidates[len(middle_candidates) // 2])
    return runtime_map

def ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_scalar(value: float, device: torch.device) -> float:
    if not ddp_enabled():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _clone_backbone_out(backbone_out: dict) -> dict:
    # Shallow copy is enough: tensors are immutable for our usage here.
    return {k: v for k, v in backbone_out.items()}


def _precompute_backbone_out(core_model, batch, forward_backbone_per_frame: bool = False):
    if forward_backbone_per_frame:
        return {"backbone_fpn": None, "vision_pos_enc": None}
    if core_model.training or not core_model.forward_backbone_per_frame_for_eval:
        return core_model.forward_image(batch.flat_img_batch)
    return {"backbone_fpn": None, "vision_pos_enc": None}


def _outputs_from_tracking_dict(output_dict: dict, num_frames: int):
    all_frame_outputs = {}
    all_frame_outputs.update(output_dict["cond_frame_outputs"])
    all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
    all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
    # Keep behavior aligned with SAM2Train.forward_tracking.
    all_frame_outputs = [
        {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
    ]
    return all_frame_outputs


def _forward_tracking_iterative(
    core_model,
    backbone_out,
    batch,
    output_dict=None,
    processing_order=None,
):
    img_feats_already_computed = backbone_out["backbone_fpn"] is not None
    if img_feats_already_computed:
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = core_model._prepare_backbone_features(backbone_out)

    num_frames = backbone_out["num_frames"]
    init_cond_frames = backbone_out["init_cond_frames"]
    frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]

    if output_dict is None:
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
    if processing_order is None:
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

    for stage_id in processing_order:
        img_ids = batch.flat_obj_to_img_idx[stage_id]
        if img_feats_already_computed:
            current_vision_feats = [x[:, img_ids] for x in vision_feats]
            current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
        else:
            (
                _,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
            ) = core_model._prepare_backbone_features_per_frame(
                batch.flat_img_batch, img_ids
            )

        current_out = core_model.track_step(
            frame_idx=stage_id,
            is_init_cond_frame=stage_id in init_cond_frames,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
            mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
            gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
        )
        add_output_as_cond_frame = stage_id in init_cond_frames or (
            core_model.add_all_frames_to_correct_as_cond
            and stage_id in frames_to_add_correction_pt
        )
        if add_output_as_cond_frame:
            output_dict["cond_frame_outputs"][stage_id] = current_out
            output_dict["non_cond_frame_outputs"].pop(stage_id, None)
        else:
            output_dict["non_cond_frame_outputs"][stage_id] = current_out
            output_dict["cond_frame_outputs"].pop(stage_id, None)

    return output_dict


def _append_middle_prompts_inplace(core_model, base_backbone_out, backbone_stage1, batch):
    # Build stage-2 prompt layout, then append only the new middle prompts to stage-1 backbone_out.
    stage2_backbone = core_model.prepare_prompt_inputs(
        _clone_backbone_out(base_backbone_out), batch
    )
    old_init = set(backbone_stage1["init_cond_frames"])
    new_init = set(stage2_backbone["init_cond_frames"])
    middle_frames = sorted(new_init - old_init)

    for t in middle_frames:
        if t in stage2_backbone["mask_inputs_per_frame"]:
            backbone_stage1["mask_inputs_per_frame"][t] = stage2_backbone["mask_inputs_per_frame"][t]

    merged_init = sorted(old_init | set(middle_frames))
    backbone_stage1["init_cond_frames"] = merged_init
    num_frames = int(backbone_stage1["num_frames"])
    backbone_stage1["frames_not_in_init_cond"] = [t for t in range(num_frames) if t not in set(merged_init)]
    return backbone_stage1, middle_frames


def is_main_process() -> bool:
    if not ddp_enabled():
        return True
    return dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def save_ckpt(path: Path, epoch: int, model, optimizer, scheduler, scaler, best_val_dice: float):
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_dice": best_val_dice,
        },
        str(path),
    )


def load_ckpt(path: Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    state = torch.load(str(path), map_location=map_location)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    unwrap_model(model).load_state_dict(model_state, strict=False)

    if optimizer is not None and isinstance(state, dict) and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and isinstance(state, dict) and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and isinstance(state, dict) and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    last_epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
    best_val_dice = float(state.get("best_val_dice", -1.0)) if isinstance(state, dict) else -1.0
    return last_epoch, best_val_dice


def resolve_best_ckpt_from_train_output(train_output_root: Path) -> Path:
    best_fold_txt = train_output_root / "best_fold.txt"
    if best_fold_txt.exists():
        content = best_fold_txt.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"best_ckpt:\s*(.+)", content)
        if m:
            p = Path(m.group(1).strip())
            if p.exists():
                return p

    summary_csv = train_output_root / "cv_summary.csv"
    if summary_csv.exists():
        best_row = None
        try:
            with open(summary_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "best_val_dice" not in row or "best_ckpt" not in row:
                        continue
                    try:
                        score = float(row["best_val_dice"])
                    except Exception:
                        continue
                    if best_row is None or score > best_row[0]:
                        best_row = (score, row["best_ckpt"])
            if best_row is not None:
                p = Path(best_row[1].strip())
                if p.exists():
                    return p
        except Exception:
            pass

    candidates = sorted(train_output_root.glob("fold_*/checkpoints/best.pth"))
    if len(candidates) > 0:
        return candidates[0]

    raise FileNotFoundError(f"Cannot resolve best checkpoint from TrainResults: {train_output_root}")


def run_epoch_two_pass(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    amp_dtype,
    train_mode: bool,
    stage1_loss_weight: float,
    stage2_loss_weight: float,
    forward_backbone_per_frame: bool = False,
    empty_cache_every: int = 0,
):
    model.train(train_mode)
    total_loss = 0.0
    total_dice = 0.0
    n_batch = 0

    core_model = unwrap_model(model)

    for batch_idx, batch in enumerate(loader, start=1):
        if train_mode:
            # Free previous step gradients before forward to reduce peak memory.
            optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device, non_blocking=True)
        base_backbone_out = _precompute_backbone_out(
            core_model,
            batch,
            forward_backbone_per_frame=forward_backbone_per_frame,
        )

        # -------------------------
        # Pass-1: boundary only
        # -------------------------
        core_model.clear_runtime_middle_prompt_map()
        backbone_stage1 = core_model.prepare_prompt_inputs(
            _clone_backbone_out(base_backbone_out), batch
        )

        stage1_needs_grad = stage1_loss_weight > 0.0
        grad_ctx = nullcontext() if stage1_needs_grad else torch.no_grad()
        with grad_ctx:
            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda"),
                dtype=amp_dtype,
            ):
                output_dict_stage1 = _forward_tracking_iterative(
                    core_model, backbone_stage1, batch
                )
                outputs_stage1 = _outputs_from_tracking_dict(
                    output_dict_stage1, backbone_stage1["num_frames"]
                )
                if stage1_needs_grad:
                    loss_stage1 = loss_fn(outputs_stage1, batch.masks)
                else:
                    loss_stage1 = torch.zeros((), device=device, dtype=torch.float32)

        runtime_middle_map = select_rule_middle_from_gt(
            batch_masks=batch.masks,
            unique_objects_identifier=batch.metadata.unique_objects_identifier,
        )

        # -------------------------
        # Pass-2 (iterative only): continue on stage-1 memory and inject middle prompt
        # -------------------------
        core_model.set_runtime_middle_prompt_map(runtime_middle_map)

        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda"),
            dtype=amp_dtype,
        ):
            backbone_stage2, middle_frames = _append_middle_prompts_inplace(
                core_model, base_backbone_out, backbone_stage1, batch
            )
            processing_order = middle_frames + [
                t for t in backbone_stage2["frames_not_in_init_cond"]
                if t not in set(middle_frames)
            ]
            output_dict_stage2 = _forward_tracking_iterative(
                core_model,
                backbone_stage2,
                batch,
                output_dict=output_dict_stage1,
                processing_order=processing_order
            )
            outputs_stage2 = _outputs_from_tracking_dict(
                output_dict_stage2, backbone_stage2["num_frames"]
            )
            loss_stage2 = loss_fn(outputs_stage2, batch.masks)
            loss_raw = stage1_loss_weight * loss_stage1 + stage2_loss_weight * loss_stage2
            loss = loss_raw / (1.0 + loss_raw)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_dice += compute_batch_volume_dice(outputs_stage2, batch.masks)
        n_batch += 1

        core_model.clear_runtime_middle_prompt_map()
        # Release large per-batch references as early as possible.
        del outputs_stage2, output_dict_stage2, backbone_stage2
        del backbone_stage1, output_dict_stage1, base_backbone_out
        if "loss_raw" in locals():
            del loss_raw
        del runtime_middle_map, loss_stage2, loss_stage1, loss
        if (
            empty_cache_every > 0
            and device.type == "cuda"
            and (batch_idx % empty_cache_every) == 0
        ):
            torch.cuda.empty_cache()

    if n_batch == 0:
        avg_loss, avg_dice = 0.0, 0.0
    else:
        avg_loss, avg_dice = total_loss / n_batch, total_dice / n_batch

    avg_loss = reduce_scalar(avg_loss, device)
    avg_dice = reduce_scalar(avg_dice, device)
    return avg_loss, avg_dice


def make_folds(patient_dirs, num_folds: int, seed: int):
    patient_dirs = list(patient_dirs)
    if num_folds <= 1:
        # Single-fold mode: one holdout split (train/val are non-overlapping).
        rng = np.random.RandomState(seed)
        idx = np.arange(len(patient_dirs))
        rng.shuffle(idx)
        val_count = max(1, int(round(len(patient_dirs) * 0.2)))
        if val_count >= len(patient_dirs):
            val_count = max(1, len(patient_dirs) - 1)
        val_idx = set(idx[:val_count].tolist())
        train = [patient_dirs[i] for i in range(len(patient_dirs)) if i not in val_idx]
        val = [patient_dirs[i] for i in range(len(patient_dirs)) if i in val_idx]
        return [(train, val)]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(patient_dirs))
    rng.shuffle(idx)
    folds = np.array_split(idx, num_folds)
    out = []
    for fold_idx in range(num_folds):
        val_idx = set(folds[fold_idx].tolist())
        train = [patient_dirs[i] for i in range(len(patient_dirs)) if i not in val_idx]
        val = [patient_dirs[i] for i in range(len(patient_dirs)) if i in val_idx]
        out.append((train, val))
    return out


def main():
    parser = argparse.ArgumentParser("SAM2 upper/lower -> rule middle iterative mask finetuning")
    parser.add_argument("--train-root", type=Path, required=True, help="Directory containing train patient folders")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for folds/checkpoints/logs")
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=DEFAULT_MODEL_CFG,
        help="SAM2 model config in Hydra (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)",
    )
    parser.add_argument(
        "--init-train-output-root",
        type=Path,
        required=True,
        help="External TrainResults root; script auto-resolves best fold checkpoint for shared stage-1/stage-2 initialization.",
    )
    parser.add_argument("--image-name", type=str, default="image.nii.gz")
    parser.add_argument("--mask-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--vision-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eta-min-factor", type=float, default=0.1)
    parser.add_argument("--freeze-image-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-image-encoder", dest="freeze_image_encoder", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--forward-backbone-per-frame",
        action="store_true",
        help="Compute image backbone features on demand per frame (lower GPU memory, slower).",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N batches (0 disables). Useful for fragmentation OOM.",
    )
    parser.add_argument("--stage1-loss-weight", type=float, default=0.0)
    parser.add_argument("--stage2-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each fold from fold_x/checkpoints/last.pth when available.",
    )
    parser.add_argument(
        "--resume-folds",
        type=str,
        default="",
        help="Optional comma-separated fold ids to resume only, e.g. '2,3'. Empty means all folds.",
    )
    args = parser.parse_args()

    # Full-case training uses variable number of frames across patients.
    # collate_fn requires equal T within one batch. Therefore batch size must be 1.
    if args.batch_size != 1:
        print(
            f"[WARN] batch_size={args.batch_size} is not supported for full-case variable-length videos. "
            "Force set to 1."
        )
        args.batch_size = 1

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = 0

    if not args.train_root.exists():
        raise FileNotFoundError(f"train root not found: {args.train_root}")
    if not args.init_train_output_root.exists():
        raise FileNotFoundError(f"init TrainResults root not found: {args.init_train_output_root}")

    init_ckpt = resolve_best_ckpt_from_train_output(args.init_train_output_root)
    if is_main_process():
        print(f"[INFO] init checkpoint resolved from {args.init_train_output_root}: {init_ckpt}")

    set_seed(args.seed)
    if use_ddp:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    patient_dirs = sorted([p for p in args.train_root.iterdir() if p.is_dir()], key=patient_sort_key)
    if len(patient_dirs) < args.num_folds:
        raise ValueError(f"patients ({len(patient_dirs)}) < num_folds ({args.num_folds})")

    if is_main_process():
        args.output_root.mkdir(parents=True, exist_ok=True)
    split_csv = args.output_root / "fold_split.csv"

    fold_defs = make_folds(patient_dirs, args.num_folds, args.seed)
    if is_main_process():
        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "split", "patient"])
            for i, (train_list, val_list) in enumerate(fold_defs):
                for p in train_list:
                    writer.writerow([i, "train", p.name])
                for p in val_list:
                    writer.writerow([i, "val", p.name])

    model_cfg_dict_template = load_model_cfg_dict_once(args.model_cfg)

    summary_rows = []
    resume_fold_set = None
    if args.resume_folds.strip():
        resume_fold_set = set(int(x.strip()) for x in args.resume_folds.split(",") if x.strip())
    for fold_idx, (train_patients, val_patients) in enumerate(fold_defs):
        if is_main_process():
            print(f"\n[Fold {fold_idx}] train={len(train_patients)} val={len(val_patients)}")
        fold_dir = args.output_root / f"fold_{fold_idx}"
        ckpt_dir = fold_dir / "checkpoints"
        log_dir = fold_dir / "logs"
        if is_main_process():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

        train_ds = UpperLowerVolumeDataset(
            train_patients,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
            enable_augment=True,
        )
        val_ds = UpperLowerVolumeDataset(
            val_patients,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
            enable_augment=False,
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, dict_key="all"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=val_sampler,
            collate_fn=lambda b: collate_fn(b, dict_key="all"),
        )

        model = build_model(
            model_cfg_dict_template=model_cfg_dict_template,
            pretrained_ckpt=init_ckpt,
            freeze_image_encoder=args.freeze_image_encoder,
            device=device,
        )
        if use_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        optimizer = build_optimizer(model, args.base_lr, args.vision_lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=args.base_lr * args.eta_min_factor,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
        loss_fn = compute_dice_ce_loss

        best_val_dice = -1.0
        best_epoch = -1
        history = []
        start_epoch = 0

        if args.resume and (resume_fold_set is None or fold_idx in resume_fold_set):
            resume_ckpt = ckpt_dir / "last.pth"
            if resume_ckpt.exists():
                start_epoch, best_val_dice = load_ckpt(
                    path=resume_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    map_location="cpu",
                )
                best_epoch = start_epoch
                if is_main_process():
                    print(
                        f"[Fold {fold_idx}] Resume from {resume_ckpt} | "
                        f"last_epoch={start_epoch} -> next_epoch={start_epoch + 1}, "
                        f"best_val_dice={best_val_dice:.6f}"
                    )
                hist_json = log_dir / "history.json"
                if hist_json.exists():
                    try:
                        with open(hist_json, "r", encoding="utf-8") as f:
                            old_history = json.load(f)
                        if isinstance(old_history, list):
                            history = old_history
                    except Exception:
                        if is_main_process():
                            print(f"[Fold {fold_idx}] WARN: failed to read history.json, continue with empty history")

        if start_epoch >= args.epochs:
            if is_main_process():
                print(
                    f"[Fold {fold_idx}] Already finished "
                    f"(last_epoch={start_epoch}, target_epochs={args.epochs}), skip training loop."
                )

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            tr_loss, tr_dice = run_epoch_two_pass(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                train_mode=True,
                stage1_loss_weight=args.stage1_loss_weight,
                stage2_loss_weight=args.stage2_loss_weight,
                forward_backbone_per_frame=args.forward_backbone_per_frame,
                empty_cache_every=args.empty_cache_every,
            )

            with torch.no_grad():
                va_loss, va_dice = run_epoch_two_pass(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    amp_dtype=amp_dtype,
                    train_mode=False,
                    stage1_loss_weight=args.stage1_loss_weight,
                    stage2_loss_weight=args.stage2_loss_weight,
                    forward_backbone_per_frame=args.forward_backbone_per_frame,
                    empty_cache_every=args.empty_cache_every,
                )

            scheduler.step(va_dice)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": tr_loss,
                    "train_dice": tr_dice,
                    "val_loss": va_loss,
                    "val_dice": va_dice,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if is_main_process():
                print(
                    f"[Fold {fold_idx}] Epoch {epoch+1}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | "
                    f"val_loss={va_loss:.4f} val_dice={va_dice:.4f}"
                )

            save_ckpt(
                ckpt_dir / "last.pth",
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_dice,
            )
            if va_dice > best_val_dice:
                best_val_dice = va_dice
                best_epoch = epoch + 1
                save_ckpt(
                    ckpt_dir / "best.pth",
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_val_dice,
                )

        if is_main_process():
            with open(log_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        if is_main_process():
            summary_rows.append(
                {
                    "fold": fold_idx,
                    "best_val_dice": best_val_dice,
                    "best_epoch": best_epoch,
                    "best_ckpt": str((ckpt_dir / "best.pth").resolve()),
                    "last_ckpt": str((ckpt_dir / "last.pth").resolve()),
                    "num_train_cases": len(train_ds),
                    "num_val_cases": len(val_ds),
                }
            )

    if is_main_process():
        summary_csv = args.output_root / "cv_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fold",
                    "best_val_dice",
                    "best_epoch",
                    "best_ckpt",
                    "last_ckpt",
                    "num_train_cases",
                    "num_val_cases",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        best_fold = max(summary_rows, key=lambda x: x["best_val_dice"])
        with open(args.output_root / "best_fold.txt", "w", encoding="utf-8") as f:
            f.write(f"best_fold: {best_fold['fold']}\n")
            f.write(f"best_val_dice: {best_fold['best_val_dice']:.6f}\n")
            f.write(f"best_epoch: {best_fold['best_epoch']}\n")
            f.write(f"best_ckpt: {best_fold['best_ckpt']}\n")

        print("\n[DONE] Training finished.")
        print(f"[DONE] CV summary: {summary_csv}")
        print(f"[DONE] Best fold: {best_fold['fold']} | val_dice={best_fold['best_val_dice']:.4f}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
