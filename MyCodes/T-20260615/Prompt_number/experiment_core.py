#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn


def _ensure_project_root_on_path():
    start = Path(__file__).resolve()
    candidates = [start.parent] + list(start.parents)
    env_root = os.environ.get("SAM2_PROJECT_ROOT", "").strip()
    if env_root:
        candidates.insert(0, Path(env_root).resolve())
    candidates.append(Path("/home/intern/ftp/wusi/SAM2"))

    for root in candidates:
        if (root / "sam2").is_dir() and (root / "training").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return


_ensure_project_root_on_path()

from sam2.modeling.lora import LoRAConfig, apply_lora, apply_qv_lora_to_fused_qkv


def uniform_prompt_indices(num_frames: int, k: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    k = int(max(1, min(int(k), num_frames)))
    if k == 1:
        return [0]
    raw = np.linspace(0, num_frames - 1, k)
    ids = [int(round(x)) for x in raw]
    ids = sorted(set(max(0, min(x, num_frames - 1)) for x in ids))
    missing = k - len(ids)
    if missing > 0:
        for x in range(num_frames):
            if x not in ids:
                ids.append(x)
                missing -= 1
                if missing == 0:
                    break
    return sorted(ids)


def sample_train_prompt_indices(num_frames: int, max_prompts: int) -> list[int]:
    hi = int(min(max_prompts, num_frames))
    lo = int(min(2, hi))
    k = random.randint(lo, hi)
    return uniform_prompt_indices(num_frames, k)


def split_prompt_masks(num_frames: int, prompt_frames: Sequence[int], device) -> tuple[torch.Tensor, torch.Tensor]:
    prompted = torch.zeros((num_frames,), dtype=torch.bool, device=device)
    for idx in prompt_frames:
        idx = int(idx)
        if 0 <= idx < num_frames:
            prompted[idx] = True
    return prompted, ~prompted


class DiceBCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        inter = (probs * targets).sum()
        dsc = (2.0 * inter + 1e-5) / (probs.sum() + targets.sum() + 1e-5)
        return 0.5 * (1.0 - dsc) + 0.5 * bce


def _stack_logits(outputs: Sequence[dict]) -> torch.Tensor:
    return torch.stack([out["pred_masks_high_res"][:, 0] for out in outputs], dim=0)


def unprompted_only_loss(
    outputs: Sequence[dict],
    gt_masks_tohw: torch.Tensor,
    prompt_frames: Sequence[int],
    criterion: DiceBCELoss | None = None,
) -> torch.Tensor:
    criterion = criterion or DiceBCELoss()
    logits_tohw = _stack_logits(outputs)
    _, unprompted = split_prompt_masks(logits_tohw.shape[0], prompt_frames, logits_tohw.device)
    if bool(unprompted.any()):
        return criterion(logits_tohw[unprompted], gt_masks_tohw[unprompted].float())
    return criterion(logits_tohw, gt_masks_tohw.float())


def dice_3d_from_logits(
    outputs: Sequence[dict],
    gt_masks_tohw: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
    threshold: float = 0.0,
) -> float:
    logits_tohw = _stack_logits(outputs)
    if frame_mask is not None:
        logits_tohw = logits_tohw[frame_mask]
        gt_masks_tohw = gt_masks_tohw[frame_mask]
    if logits_tohw.numel() == 0:
        return float("nan")
    pred = (logits_tohw > threshold).float()
    gt = gt_masks_tohw.float()
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    if float(denom.item()) == 0.0:
        return 1.0
    return float(((2.0 * inter + 1e-6) / (denom + 1e-6)).item())


def unprompted_slice_3d_dsc(outputs: Sequence[dict], gt_masks_tohw: torch.Tensor, prompt_frames: Sequence[int]) -> float:
    _, unprompted = split_prompt_masks(len(outputs), prompt_frames, gt_masks_tohw.device)
    return dice_3d_from_logits(outputs, gt_masks_tohw, unprompted)


def mean_unprompted_2d_dice(outputs: Sequence[dict], gt_masks_tohw: torch.Tensor, prompt_frames: Sequence[int]) -> float:
    logits_tohw = _stack_logits(outputs)
    _, unprompted = split_prompt_masks(logits_tohw.shape[0], prompt_frames, logits_tohw.device)
    vals = []
    for t in torch.nonzero(unprompted, as_tuple=False).flatten().tolist():
        pred = (logits_tohw[t] > 0).float()
        gt = gt_masks_tohw[t].float()
        inter = (pred * gt).sum()
        denom = pred.sum() + gt.sum()
        vals.append(float(((2.0 * inter + 1e-6) / (denom + 1e-6)).item()))
    return float(np.mean(vals)) if vals else float("nan")


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def surface_distances_mm(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Sequence[float]) -> np.ndarray:
    pred_s = _surface(pred)
    gt_s = _surface(gt)
    if not pred_s.any() or not gt_s.any():
        return np.asarray([], dtype=np.float32)
    dt_gt = ndimage.distance_transform_edt(~gt_s, sampling=spacing_zyx)
    dt_pred = ndimage.distance_transform_edt(~pred_s, sampling=spacing_zyx)
    return np.concatenate([dt_gt[pred_s], dt_pred[gt_s]]).astype(np.float32)


def hd95_asd(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Sequence[float]) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return float("inf"), float("inf")
    d = surface_distances_mm(pred, gt, spacing_zyx)
    if d.size == 0:
        return 0.0, 0.0
    return float(np.percentile(d, 95)), float(np.mean(d))


def _prob_outputs(outputs: Sequence[dict]) -> list[torch.Tensor]:
    return [torch.sigmoid(out["pred_masks_high_res"]) for out in outputs]


def fuse_bidirectional_outputs(outputs_forward: Sequence[dict], outputs_backward: Sequence[dict]) -> list[dict]:
    if len(outputs_forward) != len(outputs_backward):
        raise ValueError("forward/backward outputs must have the same length")
    fused = []
    for out_f, out_b in zip(outputs_forward, outputs_backward):
        prob = 0.5 * (_prob_outputs([out_f])[0] + _prob_outputs([out_b])[0])
        logits = torch.logit(prob.clamp(1e-4, 1.0 - 1e-4))
        out = dict(out_f)
        out["pred_masks_high_res"] = logits
        fused.append(out)
    return fused


def configure_prompt_number_trainables(
    model: nn.Module,
    lora_r: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> dict[str, int]:
    for p in model.parameters():
        p.requires_grad = False

    image_lora = apply_qv_lora_to_fused_qkv(
        model,
        LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=("qkv",),
            target_prefixes=("image_encoder",),
            freeze_base_model=False,
        ),
    )
    memory_lora = apply_lora(
        model,
        LoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=("q_proj", "v_proj"),
            target_prefixes=("memory_attention",),
            freeze_base_model=False,
        ),
    )

    for module_name in ("sam_prompt_encoder", "sam_mask_decoder"):
        module = getattr(model, module_name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "image_encoder_lora_layers": int(image_lora),
        "memory_attention_lora_layers": int(memory_lora),
        "trainable_params": int(trainable),
        "total_params": int(total),
    }


@dataclass
class ValidationResult:
    k: int
    unprompted_slice_3d_dsc: float
    whole_volume_3d_dsc: float
    unprompted_mean_2d_dice: float


def checkpoint_metric(results: Iterable[ValidationResult]) -> float:
    vals = [r.unprompted_slice_3d_dsc for r in results if math.isfinite(r.unprompted_slice_3d_dsc)]
    return float(np.mean(vals)) if vals else float("-inf")
