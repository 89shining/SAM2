#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2 Revision - Point / Disk (TEST VERSION)

Clinical deployment logic for the small.pth configuration.

CLI
---
SAM2_Revision_Point.exe ^
  -nii input_ct.nii.gz ^
  -initial initial_ctv.nii.gz ^
  -pos pos_points.json ^
  -neg neg_points.json ^
  -out prediction.nii.gz

JSON
----
{
  "points": [
    {"slice_index": 30, "x": 245.2, "y": 310.7},
    {"slice_index": 35, "x": 260.1, "y": 298.4}
  ]
}

Coordinate convention
---------------------
- slice_index: ORIGINAL full-size CT axial z index (integer)
- x: ORIGINAL full-size CT voxel column index (float)
- y: ORIGINAL full-size CT voxel row index (float)

Locked small.pth configuration
------------------------------
POS:
    point -> complete 2 mm physical XY disk
    SAM2 probability threshold >= 0.5
    physical propagation gate m = 30 mm
    final role: add outside Initial CTV

NEG:
    point -> complete 10 mm physical XY disk
    SAM2 probability threshold < 0.5
    physical propagation gate m = 15 mm
    final role: remove inside Initial CTV

Important
---------
1. No training-time prompt simulation is used clinically:
   NO erosion, NO dilation, NO connected-component filtering,
   NO Top-3 filtering, NO area filtering, NO EDT-center search.

2. The point itself is the disk center.

3. The generated disk is NOT clipped by Initial CTV, error masks,
   components, GT, body, POS/NEG semantic domains, etc.
   It is clipped only by the image FOV.

4. All input/output NIfTI volumes are ORIGINAL full-size volumes.

5. Internally only the SI slab of the Initial CTV is processed:
       z_min = first non-empty Initial CTV slice
       z_max = last non-empty Initial CTV slice
   There is NO XY crop.

6. Points outside the Initial CTV SI slab are ignored.
   Their x/y/z coordinates are still interpreted in the ORIGINAL full-size CT.

7. Every run starts from the same Initial CTV.
   A previous revision prediction must not be fed back as Initial CTV unless
   the calling application deliberately chooses to do so.

Final deployment target
-----------------------
SAM2_Revision_Point/
├── SAM2_Revision_Point.exe
├── _internal/
└── checkpoint/
    ├── sam2.1_hiera_small.pt
    ├── pos_best.pth     # small.pth POS r=2 mm model
    └── neg_best.pth     # small.pth NEG r=10 mm model
"""

from __future__ import annotations

import argparse
import json
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
from scipy import ndimage
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


# =============================================================================
# Locked deployment configuration
# =============================================================================

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

INPUT_SIZE = 512
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 400.0

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# small.pth validation-locked clinical configuration
POS_DISK_RADIUS_MM = 2.0
NEG_DISK_RADIUS_MM = 10.0
POS_GATE_MM = 30.0
NEG_GATE_MM = 15.0

POS_THRESHOLD = 0.5
NEG_RETENTION_THRESHOLD = 0.5

AMP_DTYPE = "bfloat16"
SEED = 20260724


def get_program_dir() -> Path:
    """Executable directory after PyInstaller; source directory during testing."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_program_dir()
CHECKPOINT_DIR = BASE_DIR / "checkpoint"

BASE_CHECKPOINT = CHECKPOINT_DIR / "sam2.1_hiera_small.pt"
POS_CHECKPOINT = CHECKPOINT_DIR / "pos_best.pth"
NEG_CHECKPOINT = CHECKPOINT_DIR / "neg_best.pth"


# =============================================================================
# CLI / IO
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 Revision - Point/Disk, small.pth clinical inference"
    )
    parser.add_argument(
        "-nii",
        "--niiPath",
        required=True,
        type=Path,
        help="Original full-size CT NIfTI",
    )
    parser.add_argument(
        "-initial",
        "--initialPath",
        required=True,
        type=Path,
        help="Original full-size Initial CTV NIfTI",
    )
    parser.add_argument(
        "-pos",
        "--posPoints",
        required=True,
        type=Path,
        help='POS point JSON: {"points":[{"slice_index":z,"x":x,"y":y}, ...]}',
    )
    parser.add_argument(
        "-neg",
        "--negPoints",
        required=True,
        type=Path,
        help='NEG point JSON: {"points":[{"slice_index":z,"x":x,"y":y}, ...]}',
    )
    parser.add_argument(
        "-out",
        "--outputPath",
        required=True,
        type=Path,
        help="Original full-size revised CTV NIfTI",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto = CUDA if available, otherwise CPU",
    )
    return parser.parse_args()


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
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load_windows_compatible(path: Path, map_location="cpu"):
    """
    Load trusted checkpoints created on Linux that may contain pathlib.PosixPath.
    """
    if os.name != "nt":
        return torch.load(str(path), map_location=map_location, weights_only=False)

    original_posix_path = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(str(path), map_location=map_location, weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def write_binary(path: Path, mask_zyx: np.ndarray, reference: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), useCompression=True)


# =============================================================================
# Point JSON and physical disks
# =============================================================================

@dataclass(frozen=True)
class PointPrompt:
    slice_index: int
    x: float
    y: float


def read_points_json(
    path: Path,
    full_shape_zyx: tuple[int, int, int],
    branch_name: str,
) -> list[PointPrompt]:
    """
    Read full-size voxel/index coordinates.

    x and y are allowed to be floating point values.
    slice_index must be an integer.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"{branch_name} JSON root must be an object.")
    if "points" not in payload:
        raise ValueError(f'{branch_name} JSON must contain key "points".')
    if not isinstance(payload["points"], list):
        raise ValueError(f'{branch_name} JSON "points" must be a list.')

    z_size, y_size, x_size = full_shape_zyx
    points: list[PointPrompt] = []

    for index, item in enumerate(payload["points"]):
        if not isinstance(item, dict):
            raise ValueError(f"{branch_name} point #{index} must be an object.")

        missing = {"slice_index", "x", "y"} - set(item)
        if missing:
            raise ValueError(
                f"{branch_name} point #{index} missing fields: {sorted(missing)}"
            )

        z_raw = item["slice_index"]
        # bool is an int subclass; reject it explicitly.
        if isinstance(z_raw, bool) or not isinstance(z_raw, int):
            raise ValueError(
                f"{branch_name} point #{index}: slice_index must be an integer."
            )

        try:
            x = float(item["x"])
            y = float(item["y"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{branch_name} point #{index}: x/y must be numeric."
            ) from exc

        z = int(z_raw)

        if not (0 <= z < z_size):
            raise ValueError(
                f"{branch_name} point #{index}: slice_index={z} outside "
                f"[0, {z_size - 1}]"
            )
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(
                f"{branch_name} point #{index}: x/y must be finite."
            )
        if not (0.0 <= x <= x_size - 1):
            raise ValueError(
                f"{branch_name} point #{index}: x={x} outside "
                f"[0, {x_size - 1}]"
            )
        if not (0.0 <= y <= y_size - 1):
            raise ValueError(
                f"{branch_name} point #{index}: y={y} outside "
                f"[0, {y_size - 1}]"
            )

        points.append(PointPrompt(slice_index=z, x=x, y=y))

    return points


def make_physical_disk_volume(
    shape_zyx: tuple[int, int, int],
    points: Sequence[PointPrompt],
    spacing_xyz: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    """
    Generate complete XY physical disks centered directly on clinical points.

    No semantic clipping is applied.
    The only clipping is the finite image FOV because the output array itself
    has finite [Z,Y,X] dimensions.
    """
    z_size, y_size, x_size = shape_zyx
    sx, sy, _ = map(float, spacing_xyz)

    if sx <= 0 or sy <= 0:
        raise ValueError(f"Invalid XY spacing: sx={sx}, sy={sy}")
    if radius_mm <= 0:
        raise ValueError(f"Disk radius must be positive, got {radius_mm}")

    output = np.zeros(shape_zyx, dtype=bool)

    for point in points:
        z = point.slice_index

        # Restrict computation to a bounding box only for efficiency.
        # This is NOT semantic clipping: the circle is still defined in physical mm.
        rx = radius_mm / sx
        ry = radius_mm / sy

        x0 = max(0, int(math.floor(point.x - rx)))
        x1 = min(x_size - 1, int(math.ceil(point.x + rx)))
        y0 = max(0, int(math.floor(point.y - ry)))
        y1 = min(y_size - 1, int(math.ceil(point.y + ry)))

        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = (
            ((xx.astype(np.float64) - point.x) * sx) ** 2
            + ((yy.astype(np.float64) - point.y) * sy) ** 2
            <= float(radius_mm) ** 2
        )

        output[z, y0 : y1 + 1, x0 : x1 + 1] |= disk

    return output


# =============================================================================
# Full size -> SI crop
# =============================================================================

def locked_si_range(initial_zyx: np.ndarray) -> tuple[int, int]:
    """
    Determine [z_start, z_end) ONLY from Initial CTV.
    Point prompts never expand the SI slab.
    """
    occupied = np.flatnonzero(initial_zyx.reshape(initial_zyx.shape[0], -1).any(axis=1))
    if occupied.size == 0:
        raise ValueError("Initial CTV is empty; cannot determine SI range.")
    return int(occupied[0]), int(occupied[-1]) + 1


def keep_points_inside_si(
    points: Sequence[PointPrompt],
    z_start: int,
    z_end: int,
) -> tuple[list[PointPrompt], list[PointPrompt]]:
    """Return (effective, ignored_outside_si). Coordinates remain full-size."""
    effective = [point for point in points if z_start <= point.slice_index < z_end]
    ignored = [point for point in points if not (z_start <= point.slice_index < z_end)]
    return effective, ignored


def global_points_to_crop_local(
    points: Sequence[PointPrompt],
    z_start: int,
) -> list[PointPrompt]:
    """
    Only z changes because there is NO XY crop.
    """
    return [
        PointPrompt(
            slice_index=point.slice_index - z_start,
            x=point.x,
            y=point.y,
        )
        for point in points
    ]


# =============================================================================
# Model preprocessing
# =============================================================================

def window_ct(array: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    return (
        np.clip(array.astype(np.float32), low, high) - low
    ) / (high - low)


def resize_prompt_preserve_foreground(
    prompt: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """
    Exact R-20260916 behavior:
    preserve small physical disk foreground when downsampling.
    """
    if prompt.shape[-2] >= size[0] and prompt.shape[-1] >= size[1]:
        return F.adaptive_max_pool2d(prompt, output_size=size)
    return F.interpolate(prompt, size=size, mode="nearest")


def tensor_zyx(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(1)


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
            positive_prompts=self.positive_prompts.to(
                device, non_blocking=non_blocking
            ),
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
            negative_errors=self.negative_errors.to(
                device, non_blocking=non_blocking
            ),
            negative_prompts=self.negative_prompts.to(
                device, non_blocking=non_blocking
            ),
            original_hw=self.original_hw,
        )


def build_common_inputs(
    ct_crop: np.ndarray,
    initial_crop: np.ndarray,
):
    size = (INPUT_SIZE, INPUT_SIZE)

    ct_tensor = F.interpolate(
        tensor_zyx(window_ct(ct_crop, WINDOW_CENTER, WINDOW_WIDTH)),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    initial_tensor = F.interpolate(
        tensor_zyx(initial_crop),
        size=size,
        mode="nearest",
    )

    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)

    ct_rgb = (ct_tensor.repeat(1, 3, 1, 1) - mean) / std
    initial_rgb = (initial_tensor.repeat(1, 3, 1, 1) - mean) / std

    dual_stream = torch.cat((ct_rgb, initial_rgb), dim=1).float()
    return dual_stream, initial_tensor.bool()


def build_positive_batch(
    ct_crop: np.ndarray,
    initial_crop: np.ndarray,
    pos_disk_crop: np.ndarray,
) -> PositiveBatch:
    images, initial_tensor = build_common_inputs(ct_crop, initial_crop)

    prompt_tensor = resize_prompt_preserve_foreground(
        tensor_zyx(pos_disk_crop),
        (INPUT_SIZE, INPUT_SIZE),
    )

    # In clinical inference GT/error masks are unavailable and unnecessary.
    # These fields exist only for SAM2Train-compatible batch structure.
    return PositiveBatch(
        images=images,
        targets=initial_tensor,
        true_masks=initial_tensor,
        nnunet_masks=initial_tensor,
        missed_masks=torch.zeros_like(initial_tensor),
        positive_prompts=prompt_tensor.float(),
        original_hw=(int(ct_crop.shape[1]), int(ct_crop.shape[2])),
    )


def build_negative_batch(
    ct_crop: np.ndarray,
    initial_crop: np.ndarray,
    neg_disk_crop: np.ndarray,
) -> NegativeBatch:
    images, initial_tensor = build_common_inputs(ct_crop, initial_crop)

    prompt_tensor = resize_prompt_preserve_foreground(
        tensor_zyx(neg_disk_crop),
        (INPUT_SIZE, INPUT_SIZE),
    )

    return NegativeBatch(
        images=images,
        targets=initial_tensor,
        true_masks=initial_tensor,
        nnunet_masks=initial_tensor,
        negative_errors=torch.zeros_like(initial_tensor),
        negative_prompts=prompt_tensor.float(),
        original_hw=(int(ct_crop.shape[1]), int(ct_crop.shape[2])),
    )


# =============================================================================
# Exact dual-stream SAM2 / LoRA structures
# =============================================================================

class PositiveDualStreamSAM2Train(SAM2Train):
    """POS architecture matching fullmask_pos.py."""

    def __init__(self, *args, fusion_levels: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial_gate = math.log(0.1 / 0.9)
        self.dual_stream_feature_gates = nn.Parameter(
            torch.full(
                (int(fusion_levels),),
                initial_gate,
                dtype=torch.float32,
            )
        )

    def forward_image(self, img_batch: torch.Tensor) -> dict:
        if img_batch.ndim != 4 or img_batch.shape[1] != 6:
            raise ValueError(
                "PositiveDualStreamSAM2Train expects [N,6,H,W], got "
                f"{tuple(img_batch.shape)}"
            )

        ct_output = super().forward_image(img_batch[:, :3])
        initial_output = super().forward_image(img_batch[:, 3:])

        ct_fpn = ct_output["backbone_fpn"]
        initial_fpn = initial_output["backbone_fpn"]

        if len(ct_fpn) != len(initial_fpn):
            raise RuntimeError("CT and Initial-CTV FPN lengths differ.")
        if len(ct_fpn) != self.dual_stream_feature_gates.numel():
            raise RuntimeError(
                f"Unexpected FPN levels: {len(ct_fpn)} vs "
                f"{self.dual_stream_feature_gates.numel()}"
            )

        fused = dict(ct_output)
        gates = torch.sigmoid(self.dual_stream_feature_gates)

        fused["backbone_fpn"] = [
            ct_feature
            + gates[level].to(ct_feature.dtype) * initial_feature
            for level, (ct_feature, initial_feature)
            in enumerate(zip(ct_fpn, initial_fpn))
        ]

        if (
            "vision_features" in ct_output
            and "vision_features" in initial_output
        ):
            fused["vision_features"] = (
                ct_output["vision_features"]
                + gates[-1].to(ct_output["vision_features"].dtype)
                * initial_output["vision_features"]
            )

        fused["vision_pos_enc"] = ct_output["vision_pos_enc"]
        return fused


class NegativeDualStreamSAM2Train(SAM2Train):
    """NEG architecture matching fullmask_neg.py."""

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
        initial_output = super().forward_image(images[:, 3:])

        gates = torch.sigmoid(self.dual_stream_feature_gates)
        ct_fpn = ct_output["backbone_fpn"]
        initial_fpn = initial_output["backbone_fpn"]

        if len(ct_fpn) != 3 or len(initial_fpn) != 3:
            raise RuntimeError("Expected three SAM2 FPN levels.")

        output = dict(ct_output)
        output["backbone_fpn"] = [
            ct + gates[level].to(ct.dtype) * prior
            for level, (ct, prior)
            in enumerate(zip(ct_fpn, initial_fpn))
        ]
        output["vision_pos_enc"] = ct_output["vision_pos_enc"]
        return output


def load_model_config() -> dict:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_module("sam2", version_base="1.2"):
        config = compose(config_name=MODEL_CONFIG)

    return OmegaConf.to_container(config.model, resolve=True)


def install_lora(model: nn.Module) -> dict[str, int]:
    """
    Exact LoRA layout used in R-20260916:
    - image encoder fused qkv: Q/V LoRA
    - memory attention: q_proj + v_proj
    - memory encoder fuser: pwconv1 + pwconv2
    - r=4, alpha=16, dropout=0.1
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
            f"image={image_count}, "
            f"memory_attention={attention_count}, "
            f"memory_encoder={encoder_count}"
        )

    return {
        "image_encoder_lora_layers": int(image_count),
        "memory_attention_lora_layers": int(attention_count),
        "memory_encoder_lora_layers": int(encoder_count),
    }


def split_model_config():
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


def load_base_state(model: nn.Module) -> None:
    checkpoint = torch_load_windows_compatible(
        BASE_CHECKPOINT,
        map_location="cpu",
    )
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=False)


def load_finetuned_state(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch_load_windows_compatible(
        checkpoint_path,
        map_location=device,
    )
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=True)


def build_positive_model(device: torch.device):
    config, image_cfg, attention_cfg, memory_cfg = split_model_config()

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

    load_base_state(model)
    stats = install_lora(model)
    model.to(device)
    load_finetuned_state(model, POS_CHECKPOINT, device)
    model.eval()

    print(f"[POS] LoRA restored: {stats}")
    return model


def build_negative_model(device: torch.device):
    config, image_cfg, attention_cfg, memory_cfg = split_model_config()

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

    load_base_state(model)
    stats = install_lora(model)
    model.to(device)
    load_finetuned_state(model, NEG_CHECKPOINT, device)
    model.eval()

    print(f"[NEG] LoRA restored: {stats}")
    return model


# =============================================================================
# Bidirectional inference
# =============================================================================

def positive_prompt_frames(batch: PositiveBatch) -> list[int]:
    nonempty = batch.positive_prompts.flatten(1).any(dim=1)
    frames = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
    return [int(frame) for frame in frames] if frames else [0]


def negative_prompt_frames(batch: NegativeBatch) -> list[int]:
    nonempty = batch.negative_prompts.flatten(1).any(dim=1)
    frames = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
    return [int(frame) for frame in frames] if frames else [0]


def positive_track_direction(
    model: PositiveDualStreamSAM2Train,
    backbone: dict,
    batch: PositiveBatch,
    prompts: Sequence[int],
    reverse: bool,
) -> list[dict]:
    _, features, positions, sizes = model._prepare_backbone_features(backbone)

    prompts = sorted(
        set(int(frame) for frame in prompts),
        reverse=reverse,
    )
    prompt_set = set(prompts)

    remaining = [
        frame
        for frame in range(batch.num_frames)
        if frame not in prompt_set
    ]
    remaining.sort(reverse=reverse)

    outputs = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    all_outputs: dict[int, dict] = {}

    for frame in prompts + remaining:
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
            # No GT information enters clinical inference.
            gt_masks=None,
            frames_to_add_correction_pt=[],
            output_dict=outputs,
            num_frames=batch.num_frames,
            track_in_reverse=reverse,
            prev_sam_mask_logits=None,
        )

        key = (
            "cond_frame_outputs"
            if frame in prompt_set
            else "non_cond_frame_outputs"
        )
        outputs[key][frame] = current
        all_outputs[frame] = current

    # Matches POS training/test code, which discards obj_ptr from returned frames.
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
    prompts: Sequence[int],
    reverse: bool,
) -> list[dict]:
    _, features, positions, sizes = model._prepare_backbone_features(backbone)

    prompts = sorted(set(prompts), reverse=reverse)
    prompt_set = set(prompts)

    remaining = [
        frame
        for frame in range(batch.num_frames)
        if frame not in prompt_set
    ]
    remaining.sort(reverse=reverse)

    outputs = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    all_outputs: dict[int, dict] = {}

    for frame in prompts + remaining:
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
            gt_masks=None,
            frames_to_add_correction_pt=[],
            output_dict=outputs,
            num_frames=batch.num_frames,
            track_in_reverse=reverse,
            prev_sam_mask_logits=None,
        )

        key = (
            "cond_frame_outputs"
            if frame in prompt_set
            else "non_cond_frame_outputs"
        )
        outputs[key][frame] = current
        all_outputs[frame] = current

    return [
        all_outputs[frame]
        for frame in range(batch.num_frames)
    ]


def fuse_bidirectional_probabilities(
    forward_outputs: Sequence[dict],
    backward_outputs: Sequence[dict],
) -> torch.Tensor:
    forward_logits = torch.stack(
        [
            output["pred_masks_high_res"][:, 0]
            for output in forward_outputs
        ],
        dim=0,
    )

    backward_logits = torch.stack(
        [
            output["pred_masks_high_res"][:, 0]
            for output in backward_outputs
        ],
        dim=0,
    )

    return 0.5 * (
        torch.sigmoid(forward_logits)
        + torch.sigmoid(backward_logits)
    )


@torch.no_grad()
def positive_probability(
    model: PositiveDualStreamSAM2Train,
    batch: PositiveBatch,
    device: torch.device,
) -> np.ndarray:
    batch = batch.to(device, non_blocking=True)
    prompts = positive_prompt_frames(batch)
    backbone = model.forward_image(batch.flat_img_batch)

    amp_dtype = (
        torch.bfloat16
        if AMP_DTYPE == "bfloat16"
        else torch.float16
    )

    with torch.cuda.amp.autocast(
        enabled=device.type == "cuda",
        dtype=amp_dtype,
    ):
        forward = positive_track_direction(
            model, backbone, batch, prompts, False
        )
        backward = positive_track_direction(
            model, backbone, batch, prompts, True
        )
        probability = fuse_bidirectional_probabilities(
            forward,
            backward,
        )

        probability = F.interpolate(
            probability.float(),
            size=batch.original_hw,
            mode="bilinear",
            align_corners=False,
        )

    return probability[:, 0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def negative_probability(
    model: NegativeDualStreamSAM2Train,
    batch: NegativeBatch,
    device: torch.device,
) -> np.ndarray:
    batch = batch.to(device, non_blocking=True)
    prompts = negative_prompt_frames(batch)
    backbone = model.forward_image(batch.flat_img_batch)

    amp_dtype = (
        torch.bfloat16
        if AMP_DTYPE == "bfloat16"
        else torch.float16
    )

    with torch.cuda.amp.autocast(
        enabled=device.type == "cuda",
        dtype=amp_dtype,
    ):
        forward = negative_track_direction(
            model, backbone, batch, prompts, False
        )
        backward = negative_track_direction(
            model, backbone, batch, prompts, True
        )
        probability = fuse_bidirectional_probabilities(
            forward,
            backward,
        )

        probability = F.interpolate(
            probability.float(),
            size=batch.original_hw,
            mode="bilinear",
            align_corners=False,
        )

    return probability[:, 0].cpu().numpy().astype(np.float32)


# =============================================================================
# Physical gate + branch semantics
# =============================================================================

def physical_gate_from_prompt(
    prompt_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    gate_mm: float,
) -> np.ndarray:
    """
    Exact testing principle:
    EDT is computed AFTER SAM2 probability inference.
    spacing order for [Z,Y,X] arrays is [sz,sy,sx].
    """
    if not prompt_zyx.any():
        return np.zeros_like(prompt_zyx, dtype=bool)

    sampling_zyx = tuple(reversed(tuple(map(float, spacing_xyz))))
    distance = ndimage.distance_transform_edt(
        ~prompt_zyx,
        sampling=sampling_zyx,
    )
    return distance <= float(gate_mm)


def apply_positive_branch(
    initial_crop: np.ndarray,
    pos_disk_crop: np.ndarray,
    probability: np.ndarray,
    gate_mask: np.ndarray,
) -> np.ndarray:
    """
    Exact small.pth test rule:
    nnunet | prompt | ((p>=0.5) & gate & ~nnunet & ~prompt)
    """
    return (
        initial_crop
        | pos_disk_crop
        | (
            (probability >= POS_THRESHOLD)
            & gate_mask
            & ~initial_crop
            & ~pos_disk_crop
        )
    )


def apply_negative_branch(
    initial_crop: np.ndarray,
    neg_disk_crop: np.ndarray,
    probability: np.ndarray,
    gate_mask: np.ndarray,
) -> np.ndarray:
    """
    Exact small.pth test rule:
    nnunet & ~(prompt | ((p<0.5) & gate & nnunet & ~prompt))
    """
    return initial_crop & ~(
        neg_disk_crop
        | (
            (probability < NEG_RETENTION_THRESHOLD)
            & gate_mask
            & initial_crop
            & ~neg_disk_crop
        )
    )


def fuse_branches(
    initial_crop: np.ndarray,
    pos_prediction: np.ndarray,
    neg_prediction: np.ndarray,
) -> np.ndarray:
    """
    Preserve POS additions; NEG may remove only original Initial-CTV voxels.
    Exact test fusion:
        fused = pos & ~(initial & ~neg)
    """
    return pos_prediction & ~(initial_crop & ~neg_prediction)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()

    required = {
        "CT": args.niiPath,
        "Initial CTV": args.initialPath,
        "POS points JSON": args.posPoints,
        "NEG points JSON": args.negPoints,
        "SAM2 base checkpoint": BASE_CHECKPOINT,
        "POS small.pth checkpoint": POS_CHECKPOINT,
        "NEG small.pth checkpoint": NEG_CHECKPOINT,
    }

    for name, path in required.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    set_seed()

    # -------------------------------------------------------------------------
    # Full-size input
    # -------------------------------------------------------------------------
    ct_image = sitk.ReadImage(str(args.niiPath))
    initial_image = sitk.ReadImage(str(args.initialPath))

    if not same_geometry(ct_image, initial_image):
        raise ValueError(
            "CT and Initial CTV geometry mismatch "
            "(size/spacing/origin/direction)."
        )

    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32, copy=False)
    initial = sitk.GetArrayFromImage(initial_image) > 0

    if ct.ndim != 3 or initial.ndim != 3:
        raise ValueError(
            f"Only 3D NIfTI is supported: CT={ct.shape}, Initial={initial.shape}"
        )
    if ct.shape != initial.shape:
        raise ValueError(
            f"CT/Initial shape mismatch: CT={ct.shape}, Initial={initial.shape}"
        )

    full_shape = tuple(int(value) for value in ct.shape)
    spacing_xyz = tuple(float(value) for value in ct_image.GetSpacing())

    # -------------------------------------------------------------------------
    # Read ORIGINAL full-size point coordinates
    # -------------------------------------------------------------------------
    pos_points_all = read_points_json(
        args.posPoints,
        full_shape,
        "POS",
    )
    neg_points_all = read_points_json(
        args.negPoints,
        full_shape,
        "NEG",
    )

    # -------------------------------------------------------------------------
    # SI lock from Initial CTV ONLY
    # -------------------------------------------------------------------------
    z_start, z_end = locked_si_range(initial)

    pos_points, pos_ignored = keep_points_inside_si(
        pos_points_all,
        z_start,
        z_end,
    )
    neg_points, neg_ignored = keep_points_inside_si(
        neg_points_all,
        z_start,
        z_end,
    )

    pos_local = global_points_to_crop_local(
        pos_points,
        z_start,
    )
    neg_local = global_points_to_crop_local(
        neg_points,
        z_start,
    )

    # -------------------------------------------------------------------------
    # SI crop only. XY is still the ORIGINAL full-size XY.
    # -------------------------------------------------------------------------
    ct_crop = ct[z_start:z_end]
    initial_crop = initial[z_start:z_end]

    crop_shape = tuple(int(value) for value in ct_crop.shape)

    # -------------------------------------------------------------------------
    # Clinical point -> complete physical disk
    # NO erosion/dilation/component/GT/Initial clipping.
    # -------------------------------------------------------------------------
    pos_disk_crop = make_physical_disk_volume(
        crop_shape,
        pos_local,
        spacing_xyz,
        POS_DISK_RADIUS_MM,
    )

    neg_disk_crop = make_physical_disk_volume(
        crop_shape,
        neg_local,
        spacing_xyz,
        NEG_DISK_RADIUS_MM,
    )

    print("==================================================")
    print("SAM2 Revision - Point / Disk (small.pth)")
    print("==================================================")
    print(f"Full CT shape [Z,Y,X]: {full_shape}")
    print(
        f"Initial CTV locked SI: z=[{z_start}, {z_end}) "
        f"({z_end-z_start} slices)"
    )
    print(f"Internal crop shape [Z,Y,X]: {crop_shape}")
    print(
        f"POS: {len(pos_points_all)} input points, "
        f"{len(pos_points)} effective, "
        f"{len(pos_ignored)} outside-SI ignored, "
        f"r={POS_DISK_RADIUS_MM:g} mm, m={POS_GATE_MM:g} mm"
    )
    print(
        f"NEG: {len(neg_points_all)} input points, "
        f"{len(neg_points)} effective, "
        f"{len(neg_ignored)} outside-SI ignored, "
        f"r={NEG_DISK_RADIUS_MM:g} mm, m={NEG_GATE_MM:g} mm"
    )
    print(f"POS disk voxels: {int(pos_disk_crop.sum())}")
    print(f"NEG disk voxels: {int(neg_disk_crop.sum())}")

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # POS branch
    # -------------------------------------------------------------------------
    if pos_disk_crop.any():
        print("\n[POS] Loading small.pth r=2 mm model...")
        pos_model = build_positive_model(device)

        pos_batch = build_positive_batch(
            ct_crop,
            initial_crop,
            pos_disk_crop,
        )

        print("[POS] Bidirectional SAM2 inference...")
        pos_prob = positive_probability(
            pos_model,
            pos_batch,
            device,
        )

        print("[POS] Applying 30 mm physical gate...")
        pos_gate = physical_gate_from_prompt(
            pos_disk_crop,
            spacing_xyz,
            POS_GATE_MM,
        )

        pos_prediction = apply_positive_branch(
            initial_crop,
            pos_disk_crop,
            pos_prob,
            pos_gate,
        )

        del pos_model, pos_batch, pos_prob
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[POS] No effective POS point -> branch is a no-op.")
        pos_prediction = initial_crop.copy()

    # -------------------------------------------------------------------------
    # NEG branch
    # -------------------------------------------------------------------------
    if neg_disk_crop.any():
        print("\n[NEG] Loading small.pth r=10 mm model...")
        neg_model = build_negative_model(device)

        neg_batch = build_negative_batch(
            ct_crop,
            initial_crop,
            neg_disk_crop,
        )

        print("[NEG] Bidirectional SAM2 inference...")
        neg_prob = negative_probability(
            neg_model,
            neg_batch,
            device,
        )

        print("[NEG] Applying 15 mm physical gate...")
        neg_gate = physical_gate_from_prompt(
            neg_disk_crop,
            spacing_xyz,
            NEG_GATE_MM,
        )

        neg_prediction = apply_negative_branch(
            initial_crop,
            neg_disk_crop,
            neg_prob,
            neg_gate,
        )

        del neg_model, neg_batch, neg_prob
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[NEG] No effective NEG point -> branch is a no-op.")
        neg_prediction = initial_crop.copy()

    # -------------------------------------------------------------------------
    # Exact POS/NEG fusion
    # -------------------------------------------------------------------------
    final_crop = fuse_branches(
        initial_crop,
        pos_prediction,
        neg_prediction,
    )

    # -------------------------------------------------------------------------
    # Restore ORIGINAL full-size output
    # SI outside Initial CTV remains zero by locked design.
    # -------------------------------------------------------------------------
    final_full = np.zeros_like(initial, dtype=bool)
    final_full[z_start:z_end] = final_crop

    write_binary(
        args.outputPath,
        final_full,
        ct_image,
    )

    additions = final_crop & ~initial_crop
    removals = initial_crop & ~final_crop

    print("\n==================================================")
    print("Revision completed")
    print("==================================================")
    print(f"Initial voxels: {int(initial.sum())}")
    print(f"Final voxels:   {int(final_full.sum())}")
    print(f"Added voxels:   {int(additions.sum())}")
    print(f"Removed voxels: {int(removals.sum())}")
    print(f"Output: {args.outputPath}")

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
