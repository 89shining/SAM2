#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2 Revision - Mask
Standalone deployment backend for PyInstaller / 3D Slicer.

Final deployment layout:
    SAM2_Revision_Mask/
    ├── SAM2_Revision_Mask.exe
    ├── _internal/
    └── checkpoint/
        ├── sam2.1_hiera_small.pt
        ├── pos_best.pth
        └── neg_best.pth

CLI:
    SAM2_Revision_Mask.exe ^
      -nii full_CT.nii.gz ^
      -initial full_initial_CTV.nii.gz ^
      -pos full_POS_prompt.nii.gz ^
      -neg full_NEG_prompt.nii.gz ^
      -out full_CTV_pred.nii.gz

Design:
- All four inputs and the output use ORIGINAL full-volume geometry.
- Revision is locked to the superior/inferior range of Initial CTV.
- POS/NEG prompts outside that SI range are ignored.
- POS only adds outside Initial CTV within the locked SI slab.
- NEG only removes/retains inside Initial CTV within the locked SI slab.
- The model internally uses the same dual-stream CT + Initial-CTV architecture
  and LoRA structure as the original POS/NEG training scripts.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from sam2.modeling.lora import (
    LoRAConfig,
    apply_lora,
    apply_qv_lora_to_fused_qkv,
)
from training.model.sam2 import SAM2Train


# ---------------------------------------------------------------------
# Fixed deployment configuration
# ---------------------------------------------------------------------
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

INPUT_SIZE = 512
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 400.0
THRESHOLD = 0.5

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

AMP_DTYPE = "bfloat16"
SEED = 20260724


def get_program_dir() -> Path:
    """Use the EXE directory after PyInstaller; source directory otherwise."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_program_dir()
CHECKPOINT_DIR = BASE_DIR / "checkpoint"

BASE_CHECKPOINT = CHECKPOINT_DIR / "sam2.1_hiera_small.pt"
POS_CHECKPOINT = CHECKPOINT_DIR / "pos_best.pth"
NEG_CHECKPOINT = CHECKPOINT_DIR / "neg_best.pth"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 Revision - Mask (full-size IO, SI-locked revision)"
    )
    parser.add_argument("-nii", required=True, type=Path, help="Original full-size CT NIfTI")
    parser.add_argument("-initial", required=True, type=Path, help="Original full-size Initial CTV")
    parser.add_argument("-pos", required=True, type=Path, help="Original full-size POS mask prompt")
    parser.add_argument("-neg", required=True, type=Path, help="Original full-size NEG mask prompt")
    parser.add_argument("-out", required=True, type=Path, help="Original full-size revised CTV")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Default auto: CUDA first, CPU fallback.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load_windows_compatible(path: Path, map_location="cpu"):
    """
    Load trusted checkpoints created on Linux that may contain pathlib.PosixPath.
    """
    if os.name != "nt":
        return torch.load(str(path), map_location=map_location)

    original_posix_path = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(str(path), map_location=map_location)
    finally:
        pathlib.PosixPath = original_posix_path


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def read_binary(path: Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image) > 0
    return image, array


def write_binary(path: Path, mask_zyx: np.ndarray, reference: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path), useCompression=True)


def window_ct(array: np.ndarray, center: float, width: float) -> np.ndarray:
    low = float(center) - float(width) / 2.0
    high = float(center) + float(width) / 2.0
    return (
        np.clip(array.astype(np.float32), low, high) - low
    ) / (high - low)


def nonzero_slices(mask_zyx: np.ndarray) -> np.ndarray:
    return np.where(mask_zyx.any(axis=(1, 2)))[0]


def locked_si_range(initial_zyx: np.ndarray) -> tuple[int, int]:
    """
    Return [z_start, z_end) determined ONLY from Initial CTV.
    Prompts cannot expand this range.
    """
    slices = nonzero_slices(initial_zyx)
    if slices.size == 0:
        raise ValueError("Initial CTV is empty; cannot determine SI range.")
    return int(slices.min()), int(slices.max()) + 1


# ---------------------------------------------------------------------
# Inference batch containers
# ---------------------------------------------------------------------
@dataclass
class PositiveBatch:
    images: torch.Tensor
    targets: torch.Tensor
    true_masks: torch.Tensor
    nnunet_masks: torch.Tensor
    missed_masks: torch.Tensor
    positive_prompts: torch.Tensor
    original_hw: tuple[int, int]

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])

    @property
    def flat_img_batch(self) -> torch.Tensor:
        return self.images

    @property
    def flat_obj_to_img_idx(self) -> torch.Tensor:
        return torch.arange(
            self.num_frames,
            dtype=torch.long,
            device=self.images.device,
        ).unsqueeze(1)

    @property
    def masks(self) -> torch.Tensor:
        return self.targets

    def to(self, device: torch.device, non_blocking: bool = False) -> "PositiveBatch":
        return PositiveBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            true_masks=self.true_masks.to(device, non_blocking=non_blocking),
            nnunet_masks=self.nnunet_masks.to(device, non_blocking=non_blocking),
            missed_masks=self.missed_masks.to(device, non_blocking=non_blocking),
            positive_prompts=self.positive_prompts.to(device, non_blocking=non_blocking),
            original_hw=self.original_hw,
        )


@dataclass
class NegativeBatch:
    images: torch.Tensor
    targets: torch.Tensor
    true_masks: torch.Tensor
    nnunet_masks: torch.Tensor
    negative_errors: torch.Tensor
    negative_prompts: torch.Tensor
    original_hw: tuple[int, int]

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])

    @property
    def flat_img_batch(self) -> torch.Tensor:
        return self.images

    @property
    def flat_obj_to_img_idx(self) -> torch.Tensor:
        return torch.arange(
            self.num_frames,
            device=self.images.device,
        ).unsqueeze(1)

    def to(self, device: torch.device, non_blocking: bool = False) -> "NegativeBatch":
        return NegativeBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            true_masks=self.true_masks.to(device, non_blocking=non_blocking),
            nnunet_masks=self.nnunet_masks.to(device, non_blocking=non_blocking),
            negative_errors=self.negative_errors.to(device, non_blocking=non_blocking),
            negative_prompts=self.negative_prompts.to(device, non_blocking=non_blocking),
            original_hw=self.original_hw,
        )


# ---------------------------------------------------------------------
# Dual-stream models
# These reproduce the original POS and NEG training architectures.
# ---------------------------------------------------------------------
class PositiveDualStreamSAM2Train(SAM2Train):
    """Exact POS dual-stream CT + Initial-CTV FPN fusion architecture."""

    def __init__(self, *args, fusion_levels: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial_gate = math.log(0.1 / 0.9)
        self.dual_stream_feature_gates = nn.Parameter(
            torch.full((int(fusion_levels),), initial_gate, dtype=torch.float32)
        )

    def forward_image(self, img_batch: torch.Tensor) -> dict:
        if img_batch.ndim != 4 or img_batch.shape[1] != 6:
            raise ValueError(
                "PositiveDualStreamSAM2Train expects [N,6,H,W], got "
                f"{tuple(img_batch.shape)}"
            )

        ct_output = super().forward_image(img_batch[:, :3])
        nnunet_output = super().forward_image(img_batch[:, 3:])

        ct_fpn = ct_output["backbone_fpn"]
        nnunet_fpn = nnunet_output["backbone_fpn"]

        if len(ct_fpn) != len(nnunet_fpn):
            raise RuntimeError("CT and Initial-CTV FPN outputs have different lengths.")
        if len(ct_fpn) != self.dual_stream_feature_gates.numel():
            raise RuntimeError(
                f"Unexpected FPN level count: {len(ct_fpn)} vs "
                f"{self.dual_stream_feature_gates.numel()}"
            )

        fused = dict(ct_output)
        gates = torch.sigmoid(self.dual_stream_feature_gates)

        fused["backbone_fpn"] = [
            ct_feature + gates[level].to(ct_feature.dtype) * prior_feature
            for level, (ct_feature, prior_feature) in enumerate(
                zip(ct_fpn, nnunet_fpn)
            )
        ]

        if "vision_features" in ct_output and "vision_features" in nnunet_output:
            fused["vision_features"] = (
                ct_output["vision_features"]
                + gates[-1].to(ct_output["vision_features"].dtype)
                * nnunet_output["vision_features"]
            )

        fused["vision_pos_enc"] = ct_output["vision_pos_enc"]
        return fused


class NegativeDualStreamSAM2Train(SAM2Train):
    """Exact NEG dual-stream CT + Initial-CTV FPN fusion architecture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial_gate = math.log(0.1 / 0.9)
        self.dual_stream_feature_gates = nn.Parameter(
            torch.full((3,), initial_gate)
        )

    def forward_image(self, images: torch.Tensor) -> dict:
        if images.ndim != 4 or images.shape[1] != 6:
            raise ValueError(f"Expected [N,6,H,W], got {tuple(images.shape)}")

        ct_output = super().forward_image(images[:, :3])
        nnunet_output = super().forward_image(images[:, 3:])

        gates = torch.sigmoid(self.dual_stream_feature_gates)
        ct_fpn = ct_output["backbone_fpn"]
        nnunet_fpn = nnunet_output["backbone_fpn"]

        if len(ct_fpn) != 3 or len(nnunet_fpn) != 3:
            raise RuntimeError("Expected three SAM2 FPN levels.")

        output = dict(ct_output)
        output["backbone_fpn"] = [
            ct + gates[level].to(ct.dtype) * prior
            for level, (ct, prior) in enumerate(zip(ct_fpn, nnunet_fpn))
        ]
        output["vision_pos_enc"] = ct_output["vision_pos_enc"]
        return output


# ---------------------------------------------------------------------
# SAM2 config + LoRA installation
# ---------------------------------------------------------------------
def load_model_config() -> dict:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_module("sam2", version_base="1.2"):
        config = compose(config_name=MODEL_CONFIG)

    return OmegaConf.to_container(config.model, resolve=True)


def install_lora(model: nn.Module) -> dict[str, int]:
    """
    Recreate the exact LoRA structure used during training:
      - image_encoder: qkv Q/V LoRA
      - memory_attention: q_proj / v_proj LoRA
      - memory_encoder.fuser: pwconv1 / pwconv2 LoRA
    Prompt encoder + mask decoder are present as trainable full modules in
    the training checkpoint, and dual_stream_feature_gates are retained.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    common = dict(
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        freeze_base_model=False,
    )

    image_count = apply_qv_lora_to_fused_qkv(
        model,
        LoRAConfig(
            target_modules=("qkv",),
            target_prefixes=("image_encoder",),
            **common,
        ),
    )
    attention_count = apply_lora(
        model,
        LoRAConfig(
            target_modules=("q_proj", "v_proj"),
            target_prefixes=("memory_attention",),
            **common,
        ),
    )
    encoder_count = apply_lora(
        model,
        LoRAConfig(
            target_modules=("pwconv1", "pwconv2"),
            target_prefixes=("memory_encoder.fuser",),
            **common,
        ),
    )

    for module in (model.sam_prompt_encoder, model.sam_mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad = True

    model.dual_stream_feature_gates.requires_grad = True

    if min(image_count, attention_count, encoder_count) == 0:
        raise RuntimeError(
            "LoRA installation failed: "
            f"image={image_count}, memory_attention={attention_count}, "
            f"memory_encoder={encoder_count}"
        )

    return {
        "image_encoder_lora_layers": int(image_count),
        "memory_attention_lora_layers": int(attention_count),
        "memory_encoder_lora_layers": int(encoder_count),
    }


def build_base_config_parts():
    config = load_model_config()
    config["image_size"] = INPUT_SIZE
    config["freeze_image_encoder"] = False
    config["use_mask_input_as_output_without_sam"] = False
    config["pred_obj_scores"] = False
    config["fixed_no_obj_ptr"] = False
    config["multimask_output_in_sam"] = False
    config["multimask_output_for_tracking"] = False

    image_encoder_config = config.pop("image_encoder")
    memory_attention_config = config.pop("memory_attention")
    memory_encoder_config = config.pop("memory_encoder")
    config.pop("_target_", None)

    return (
        config,
        image_encoder_config,
        memory_attention_config,
        memory_encoder_config,
    )


def load_finetuned_state(model: nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch_load_windows_compatible(checkpoint_path, map_location=device)
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=True)


def build_positive_model(device: torch.device):
    config, image_cfg, attention_cfg, memory_cfg = build_base_config_parts()

    model = PositiveDualStreamSAM2Train(
        image_encoder=instantiate(image_cfg, _recursive_=True),
        memory_attention=instantiate(attention_cfg, _recursive_=True),
        memory_encoder=instantiate(memory_cfg, _recursive_=True),
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
        fusion_levels=3,
        **config,
    )

    base = torch_load_windows_compatible(BASE_CHECKPOINT, map_location="cpu")
    base_state = base["model"] if isinstance(base, dict) and "model" in base else base
    model.load_state_dict(base_state, strict=False)

    lora_stats = install_lora(model)
    model.to(device)
    load_finetuned_state(model, POS_CHECKPOINT, device)
    model.eval()

    print(f"POS LoRA restored: {lora_stats}")
    return model


def build_negative_model(device: torch.device):
    config, image_cfg, attention_cfg, memory_cfg = build_base_config_parts()

    model = NegativeDualStreamSAM2Train(
        image_encoder=instantiate(image_cfg, _recursive_=True),
        memory_attention=instantiate(attention_cfg, _recursive_=True),
        memory_encoder=instantiate(memory_cfg, _recursive_=True),
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        num_frames_to_correct_for_train=1,
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        num_correction_pt_per_frame=0,
        rand_init_cond_frames_for_train=False,
        rand_init_cond_frames_for_eval=False,
        **config,
    )

    base = torch_load_windows_compatible(BASE_CHECKPOINT, map_location="cpu")
    base_state = base["model"] if isinstance(base, dict) and "model" in base else base
    model.load_state_dict(base_state, strict=False)

    lora_stats = install_lora(model)
    model.to(device)
    load_finetuned_state(model, NEG_CHECKPOINT, device)
    model.eval()

    print(f"NEG LoRA restored: {lora_stats}")
    return model


# ---------------------------------------------------------------------
# Input tensor construction
# ---------------------------------------------------------------------
def to_4d(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(1)


def prepare_common_tensors(
    ct_zyx: np.ndarray,
    initial_zyx: np.ndarray,
    prompt_zyx: np.ndarray,
):
    ct = window_ct(ct_zyx, WINDOW_CENTER, WINDOW_WIDTH)
    size = (INPUT_SIZE, INPUT_SIZE)

    ct_tensor = F.interpolate(
        to_4d(ct),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    initial_tensor = F.interpolate(
        to_4d(initial_zyx),
        size=size,
        mode="nearest",
    )
    prompt_tensor = F.interpolate(
        to_4d(prompt_zyx),
        size=size,
        mode="nearest",
    )

    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)

    ct_rgb = (ct_tensor.repeat(1, 3, 1, 1) - mean) / std
    initial_rgb = (initial_tensor.repeat(1, 3, 1, 1) - mean) / std

    dual_stream = torch.cat((ct_rgb, initial_rgb), dim=1).float()
    return dual_stream, initial_tensor.bool(), prompt_tensor.float()


def build_positive_batch(
    ct_zyx: np.ndarray,
    initial_zyx: np.ndarray,
    prompt_zyx: np.ndarray,
) -> PositiveBatch:
    images, initial_tensor, prompt_tensor = prepare_common_tensors(
        ct_zyx, initial_zyx, prompt_zyx
    )
    return PositiveBatch(
        images=images,
        targets=initial_tensor,
        true_masks=initial_tensor,
        nnunet_masks=initial_tensor,
        missed_masks=prompt_tensor.bool(),
        positive_prompts=prompt_tensor,
        original_hw=(int(ct_zyx.shape[1]), int(ct_zyx.shape[2])),
    )


def build_negative_batch(
    ct_zyx: np.ndarray,
    initial_zyx: np.ndarray,
    prompt_zyx: np.ndarray,
) -> NegativeBatch:
    images, initial_tensor, prompt_tensor = prepare_common_tensors(
        ct_zyx, initial_zyx, prompt_zyx
    )
    return NegativeBatch(
        images=images,
        targets=initial_tensor,
        true_masks=initial_tensor,
        nnunet_masks=initial_tensor,
        negative_errors=prompt_tensor.bool(),
        negative_prompts=prompt_tensor,
        original_hw=(int(ct_zyx.shape[1]), int(ct_zyx.shape[2])),
    )


# ---------------------------------------------------------------------
# Exact bidirectional mask-prompt propagation logic
# ---------------------------------------------------------------------
def positive_prompt_frames(batch: PositiveBatch) -> list[int]:
    nonempty = batch.positive_prompts.flatten(1).any(dim=1)
    indices = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
    return [int(index) for index in indices] if indices else [0]


def negative_prompt_frames(batch: NegativeBatch) -> list[int]:
    nonempty = batch.negative_prompts.flatten(1).any(dim=1)
    indices = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
    return [int(index) for index in indices] if indices else [0]


def positive_track_direction(
    model: PositiveDualStreamSAM2Train,
    backbone: dict,
    batch: PositiveBatch,
    prompt_frames: Sequence[int],
    reverse: bool,
) -> list[dict]:
    _, features, positions, sizes = model._prepare_backbone_features(backbone)

    prompt_frames = sorted(
        set(int(frame) for frame in prompt_frames),
        reverse=reverse,
    )
    prompt_set = set(prompt_frames)

    remaining = [
        frame for frame in range(batch.num_frames)
        if frame not in prompt_set
    ]
    remaining.sort(reverse=reverse)

    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    all_outputs = {}

    for frame in prompt_frames + remaining:
        image_ids = batch.flat_obj_to_img_idx[frame]

        current = model.track_step(
            frame_idx=frame,
            is_init_cond_frame=frame in prompt_set,
            current_vision_feats=[
                feature[:, image_ids] for feature in features
            ],
            current_vision_pos_embeds=[
                position[:, image_ids] for position in positions
            ],
            feat_sizes=sizes,
            point_inputs=None,
            mask_inputs=(
                batch.positive_prompts[frame].unsqueeze(1)
                if frame in prompt_set
                else None
            ),
            gt_masks=batch.targets[frame].unsqueeze(1),
            frames_to_add_correction_pt=[],
            output_dict=output_dict,
            num_frames=batch.num_frames,
            track_in_reverse=reverse,
            prev_sam_mask_logits=None,
        )

        key = (
            "cond_frame_outputs"
            if frame in prompt_set
            else "non_cond_frame_outputs"
        )
        output_dict[key][frame] = current
        all_outputs[frame] = current

    return [
        {
            key: value
            for key, value in all_outputs[frame].items()
            if key != "obj_ptr"
        }
        for frame in range(batch.num_frames)
    ]


def negative_track_direction(
    model: NegativeDualStreamSAM2Train,
    backbone: dict,
    batch: NegativeBatch,
    prompt_frames: Sequence[int],
    reverse: bool,
) -> list[dict]:
    _, features, positions, sizes = model._prepare_backbone_features(backbone)

    prompt_frames = sorted(
        set(int(frame) for frame in prompt_frames),
        reverse=reverse,
    )
    prompt_set = set(prompt_frames)

    remaining = [
        frame for frame in range(batch.num_frames)
        if frame not in prompt_set
    ]
    remaining.sort(reverse=reverse)

    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    all_outputs = {}

    for frame in prompt_frames + remaining:
        image_ids = batch.flat_obj_to_img_idx[frame]

        current = model.track_step(
            frame_idx=frame,
            is_init_cond_frame=frame in prompt_set,
            current_vision_feats=[
                feature[:, image_ids] for feature in features
            ],
            current_vision_pos_embeds=[
                position[:, image_ids] for position in positions
            ],
            feat_sizes=sizes,
            point_inputs=None,
            mask_inputs=(
                batch.negative_prompts[frame].unsqueeze(1)
                if frame in prompt_set
                else None
            ),
            gt_masks=batch.targets[frame].unsqueeze(1),
            frames_to_add_correction_pt=[],
            output_dict=output_dict,
            num_frames=batch.num_frames,
            track_in_reverse=reverse,
            prev_sam_mask_logits=None,
        )

        key = (
            "cond_frame_outputs"
            if frame in prompt_set
            else "non_cond_frame_outputs"
        )
        output_dict[key][frame] = current
        all_outputs[frame] = current

    return [all_outputs[frame] for frame in range(batch.num_frames)]


def average_bidirectional_logits(forward: list[dict], backward: list[dict]) -> torch.Tensor:
    forward_logits = torch.stack(
        [output["pred_masks_high_res"][:, 0] for output in forward],
        dim=0,
    )
    backward_logits = torch.stack(
        [output["pred_masks_high_res"][:, 0] for output in backward],
        dim=0,
    )

    probability = 0.5 * (
        torch.sigmoid(forward_logits)
        + torch.sigmoid(backward_logits)
    )
    return torch.logit(probability.clamp(1e-4, 1 - 1e-4))


def positive_bidirectional_logits(
    model: PositiveDualStreamSAM2Train,
    batch: PositiveBatch,
) -> torch.Tensor:
    prompt_frames = positive_prompt_frames(batch)
    backbone = model.forward_image(batch.flat_img_batch)

    forward = positive_track_direction(
        model, backbone, batch, prompt_frames, False
    )
    backward = positive_track_direction(
        model, backbone, batch, prompt_frames, True
    )
    return average_bidirectional_logits(forward, backward)


def negative_bidirectional_logits(
    model: NegativeDualStreamSAM2Train,
    batch: NegativeBatch,
) -> torch.Tensor:
    prompt_frames = negative_prompt_frames(batch)
    backbone = model.forward_image(batch.flat_img_batch)

    forward = negative_track_direction(
        model, backbone, batch, prompt_frames, False
    )
    backward = negative_track_direction(
        model, backbone, batch, prompt_frames, True
    )
    return average_bidirectional_logits(forward, backward)


@torch.no_grad()
def infer_positive(
    model,
    batch: PositiveBatch,
    device: torch.device,
) -> np.ndarray:
    batch = batch.to(device)
    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16

    with torch.cuda.amp.autocast(
        enabled=device.type == "cuda",
        dtype=amp_dtype,
    ):
        logits = positive_bidirectional_logits(model, batch)

    probability = F.interpolate(
        torch.sigmoid(logits.float()),
        size=batch.original_hw,
        mode="bilinear",
        align_corners=False,
    )
    return (probability[:, 0] >= THRESHOLD).cpu().numpy()


@torch.no_grad()
def infer_negative(
    model,
    batch: NegativeBatch,
    device: torch.device,
) -> np.ndarray:
    batch = batch.to(device)
    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16

    with torch.cuda.amp.autocast(
        enabled=device.type == "cuda",
        dtype=amp_dtype,
    ):
        logits = negative_bidirectional_logits(model, batch)

    probability = F.interpolate(
        torch.sigmoid(logits.float()),
        size=batch.original_hw,
        mode="bilinear",
        align_corners=False,
    )
    return (probability[:, 0] >= THRESHOLD).cpu().numpy()


# ---------------------------------------------------------------------
# Main revision pipeline
# ---------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    required = {
        "CT": args.nii,
        "Initial CTV": args.initial,
        "POS prompt": args.pos,
        "NEG prompt": args.neg,
        "SAM2 base checkpoint": BASE_CHECKPOINT,
        "POS checkpoint": POS_CHECKPOINT,
        "NEG checkpoint": NEG_CHECKPOINT,
    }
    for name, path in required.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    set_seed()

    ct_image = sitk.ReadImage(str(args.nii))
    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)

    initial_image, initial = read_binary(args.initial)
    pos_image, pos_raw = read_binary(args.pos)
    neg_image, neg_raw = read_binary(args.neg)

    for name, image in (
        ("Initial CTV", initial_image),
        ("POS prompt", pos_image),
        ("NEG prompt", neg_image),
    ):
        if not same_geometry(ct_image, image):
            raise ValueError(
                f"Geometry mismatch: CT vs {name} "
                "(size/spacing/origin/direction must match)."
            )

    if not (ct.shape == initial.shape == pos_raw.shape == neg_raw.shape):
        raise ValueError(
            f"Shape mismatch: CT={ct.shape}, Initial={initial.shape}, "
            f"POS={pos_raw.shape}, NEG={neg_raw.shape}"
        )

    # ---------------------------------------------------------------
    # LOCKED SI range: ONLY Initial CTV defines superior/inferior bounds.
    # ---------------------------------------------------------------
    z_start, z_end = locked_si_range(initial)

    si_mask = np.zeros_like(initial, dtype=bool)
    si_mask[z_start:z_end] = True

    # First enforce POS/NEG semantic domains, then enforce the SI slab.
    pos_domain = pos_raw & ~initial
    neg_domain = neg_raw & initial

    pos_effective = pos_domain & si_mask
    neg_effective = neg_domain & si_mask

    pos_outside_si = pos_domain & ~si_mask
    neg_outside_si = neg_domain & ~si_mask

    print("==========================================")
    print("SAM2 Revision - Mask")
    print("==========================================")
    print(f"Device request: {args.device}")
    print(
        f"Locked Initial CTV SI range: z=[{z_start}, {z_end}) "
        f"({z_end-z_start} slices)"
    )
    print(
        f"POS prompt: {int(pos_raw.sum())} voxels -> "
        f"{int(pos_effective.sum())} effective; "
        f"{int(pos_outside_si.sum())} outside-SI ignored"
    )
    print(
        f"NEG prompt: {int(neg_raw.sum())} voxels -> "
        f"{int(neg_effective.sum())} effective; "
        f"{int(neg_outside_si.sum())} outside-SI ignored"
    )

    # SI-only crop. XY remains unchanged.
    ct_crop = ct[z_start:z_end]
    initial_crop = initial[z_start:z_end]
    pos_crop = pos_effective[z_start:z_end]
    neg_crop = neg_effective[z_start:z_end]

    device = resolve_device(args.device)
    print(f"Resolved device: {device}")
    print(f"Full shape [Z,H,W]: {ct.shape}")
    print(f"Revision crop [Z,H,W]: {ct_crop.shape}")

    # ---------------------------------------------------------------
    # POS branch
    # ---------------------------------------------------------------
    if pos_crop.any():
        print("Running POS branch...")
        pos_model = build_positive_model(device)
        pos_batch = build_positive_batch(ct_crop, initial_crop, pos_crop)
        pos_prediction = infer_positive(pos_model, pos_batch, device)

        del pos_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("POS branch skipped: no effective POS prompt.")
        pos_prediction = initial_crop.copy()

    # ---------------------------------------------------------------
    # NEG branch
    # ---------------------------------------------------------------
    if neg_crop.any():
        print("Running NEG branch...")
        neg_model = build_negative_model(device)
        neg_batch = build_negative_batch(ct_crop, initial_crop, neg_crop)
        neg_prediction = infer_negative(neg_model, neg_batch, device)

        del neg_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("NEG branch skipped: no effective NEG prompt.")
        neg_prediction = initial_crop.copy()

    # ---------------------------------------------------------------
    # Locked fusion
    # POS can only add OUTSIDE initial CTV.
    # NEG can only determine retention INSIDE initial CTV.
    # ---------------------------------------------------------------
    additions = pos_prediction & ~initial_crop
    retained = neg_prediction & initial_crop
    final_crop = additions | retained

    # Restore to ORIGINAL full-volume geometry.
    final_full = np.zeros_like(initial, dtype=bool)
    final_full[z_start:z_end] = final_crop

    write_binary(args.out, final_full, ct_image)

    print("==========================================")
    print("Revision completed")
    print("==========================================")
    print(f"Initial voxels: {int(initial.sum())}")
    print(f"POS additions: {int(additions.sum())}")
    print(f"NEG retained: {int(retained.sum())}")
    print(f"Final voxels: {int(final_full.sum())}")
    print(f"Output: {args.out}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
