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


def positive_slice_indices(gt_masks_tohw: torch.Tensor) -> list[int]:
    """Return frame indices whose CTV mask contains at least one foreground voxel."""
    if gt_masks_tohw.ndim < 3:
        raise ValueError(f"Expected [T,...] masks, got shape {tuple(gt_masks_tohw.shape)}")
    positive = gt_masks_tohw.reshape(gt_masks_tohw.shape[0], -1).any(dim=1)
    return torch.nonzero(positive, as_tuple=False).flatten().cpu().tolist()


def _stratified_bins(positive_indices: Sequence[int], k: int) -> list[list[int]]:
    """Split ordered CTV-positive SI locations into K non-empty contiguous strata."""
    values = sorted(set(int(x) for x in positive_indices))
    if k < 1 or k > len(values):
        raise ValueError(f"Cannot split {len(values)} positive slices into k={k} strata")
    chunks = np.array_split(np.asarray(values, dtype=np.int64), int(k))
    return [[int(x) for x in chunk.tolist()] for chunk in chunks]


def stratified_spaced_prompt_indices(
    positive_indices: Sequence[int],
    k: int,
    min_index_gap: int = 2,
    rng: random.Random | None = None,
) -> list[int]:
    """Sample one CTV-positive slice per SI stratum with an index-gap constraint."""
    rng = rng or random
    gap = max(1, int(min_index_gap))
    bins = _stratified_bins(positive_indices, int(k))
    candidates = []
    for values in bins:
        values = list(values)
        rng.shuffle(values)
        candidates.append(values)

    selected: list[int] = []

    def search(bin_idx: int) -> bool:
        if bin_idx == len(candidates):
            return True
        for z in candidates[bin_idx]:
            if selected and z - selected[-1] < gap:
                continue
            selected.append(z)
            if search(bin_idx + 1):
                return True
            selected.pop()
        return False

    if not search(0):
        raise ValueError(
            f"No SI-stratified placement for k={k}, min_index_gap={gap}, "
            f"positive_indices={sorted(set(int(x) for x in positive_indices))}"
        )
    return selected


def feasible_prompt_counts(
    positive_indices: Sequence[int],
    min_prompts: int = 1,
    max_prompts: int = 5,
    min_index_gap: int = 2,
) -> list[int]:
    """Return K values for which at least one stratified placement exists."""
    feasible = []
    for k in range(int(min_prompts), int(max_prompts) + 1):
        try:
            stratified_spaced_prompt_indices(
                positive_indices, k, min_index_gap, random.Random(0)
            )
            feasible.append(k)
        except ValueError:
            pass
    return feasible


def sample_train_prompt_indices(
    positive_indices: Sequence[int],
    min_prompts: int = 1,
    max_prompts: int = 5,
    min_index_gap: int = 2,
    rng: random.Random | None = None,
) -> list[int]:
    """Uniformly draw a feasible K, then use SI-stratified random placement."""
    rng = rng or random
    feasible = feasible_prompt_counts(
        positive_indices, min_prompts, max_prompts, min_index_gap
    )
    if not feasible:
        raise ValueError("No feasible prompt count for the supplied CTV-positive slices")
    k = rng.choice(feasible)
    return stratified_spaced_prompt_indices(positive_indices, k, min_index_gap, rng)


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
    return None


def unprompted_presence_bce(
    outputs: Sequence[dict],
    gt_masks_tohw: torch.Tensor,
    prompt_frames: Sequence[int],
) -> torch.Tensor | None:
    """Mean slice-presence BCE across all unprompted slices."""
    _, unprompted = split_prompt_masks(len(outputs), prompt_frames, gt_masks_tohw.device)
    if not bool(unprompted.any()):
        return None
    logits = torch.stack(
        [out["multistep_object_score_logits"][-1].reshape(-1)[0] for out in outputs],
        dim=0,
    )
    targets = gt_masks_tohw.reshape(gt_masks_tohw.shape[0], -1).any(dim=1).float()
    return F.binary_cross_entropy_with_logits(
        logits[unprompted].float(), targets[unprompted], reduction="mean"
    )


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
        out["directional_object_score_logits"] = (
            out_f["multistep_object_score_logits"][-1],
            out_b["multistep_object_score_logits"][-1],
        )
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

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    groups = {
        "image_encoder_qv_lora": [
            n for n in trainable_names if n.startswith("image_encoder.") and ".lora_" in n
        ],
        "memory_attention_qv_lora": [
            n for n in trainable_names if n.startswith("memory_attention.") and ".lora_" in n
        ],
        "prompt_encoder": [n for n in trainable_names if n.startswith("sam_prompt_encoder.")],
        "mask_decoder": [n for n in trainable_names if n.startswith("sam_mask_decoder.")],
    }
    missing = [name for name, values in groups.items() if not values]
    disallowed = [
        n for n in trainable_names
        if not (
            (n.startswith("image_encoder.") and ".lora_" in n)
            or (n.startswith("memory_attention.") and ".lora_" in n)
            or n.startswith("sam_prompt_encoder.")
            or n.startswith("sam_mask_decoder.")
        )
    ]
    if missing or disallowed:
        raise RuntimeError(
            "Trainable-profile assertion failed: "
            f"missing_groups={missing}, disallowed_trainable_parameters={disallowed[:20]}"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "image_encoder_lora_layers": int(image_lora),
        "memory_attention_lora_layers": int(memory_lora),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_parameter_groups": {k: len(v) for k, v in groups.items()},
    }


@dataclass
class ValidationResult:
    k: int
    unprompted_slice_3d_dsc: float
    whole_volume_3d_dsc: float
    unprompted_mean_2d_dice: float


def checkpoint_metric(results: Iterable[ValidationResult]) -> float:
    by_k = {int(r.k): float(r.unprompted_slice_3d_dsc) for r in results}
    expected = set(range(1, 6))
    if set(by_k) != expected:
        raise ValueError(f"Checkpoint metric requires exactly K=1..5, got {sorted(by_k)}")
    vals = [by_k[k] for k in range(1, 6)]
    if not all(math.isfinite(x) for x in vals):
        return float("-inf")
    return float(np.mean(vals))
