#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch

from bidirectional_tracking import bidirectional_outputs, single_direction_outputs
from experiment_core import (
    DiceBCELoss,
    ValidationResult,
    checkpoint_metric,
    dice_3d_from_logits,
    mean_unprompted_2d_dice,
    sample_train_prompt_indices,
    uniform_prompt_indices,
    unprompted_only_loss,
    unprompted_slice_3d_dsc,
)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_prompts: int,
    use_bidirectional_train: bool = False,
    grad_clip_norm: float = 1.0,
    forward_backbone_per_frame: bool = False,
    epoch: int = 0,
    seed: int = 20260616,
) -> dict[str, float]:
    model.train(True)
    criterion = DiceBCELoss().to(device)
    total_loss = 0.0
    total_unprompted_dsc = 0.0
    n_batch = 0
    core_model = unwrap_model(model)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        video_id = int(batch.metadata.unique_objects_identifier[0, 0, 0].item())
        rng_state = random.getstate()
        random.seed(int(seed) + int(epoch) * 1000003 + int(max_prompts) * 10007 + video_id)
        prompt_frames = sample_train_prompt_indices(int(batch.num_frames), max_prompts)
        random.setstate(rng_state)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
            if use_bidirectional_train:
                outputs = bidirectional_outputs(
                    core_model,
                    batch,
                    prompt_frames=prompt_frames,
                    forward_backbone_per_frame=forward_backbone_per_frame,
                )
            else:
                outputs = single_direction_outputs(
                    core_model,
                    batch,
                    prompt_frames=prompt_frames,
                    reverse=(random.random() < 0.5),
                    forward_backbone_per_frame=forward_backbone_per_frame,
                )
            loss = unprompted_only_loss(outputs, batch.masks, prompt_frames, criterion)

        if loss is None or not loss.requires_grad:
            continue

        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(grad_clip_norm),
            )
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_unprompted_dsc += unprompted_slice_3d_dsc(outputs, batch.masks, prompt_frames)
        n_batch += 1

    if n_batch == 0:
        return {"loss": 0.0, "unprompted_slice_3d_dsc": 0.0}
    return {
        "loss": total_loss / n_batch,
        "unprompted_slice_3d_dsc": total_unprompted_dsc / n_batch,
    }


@torch.no_grad()
def validate_fixed_ks(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_prompts: int,
    forward_backbone_per_frame: bool = False,
) -> tuple[list[ValidationResult], float]:
    model.train(False)
    core_model = unwrap_model(model)
    results = []

    for k in range(2, int(max_prompts) + 1):
        per_case_unprompted = []
        per_case_whole = []
        per_case_2d = []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prompt_frames = uniform_prompt_indices(int(batch.num_frames), k)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
                outputs = bidirectional_outputs(
                    core_model,
                    batch,
                    prompt_frames=prompt_frames,
                    forward_backbone_per_frame=forward_backbone_per_frame,
                )
            per_case_unprompted.append(unprompted_slice_3d_dsc(outputs, batch.masks, prompt_frames))
            per_case_whole.append(dice_3d_from_logits(outputs, batch.masks))
            per_case_2d.append(mean_unprompted_2d_dice(outputs, batch.masks, prompt_frames))

        results.append(
            ValidationResult(
                k=k,
                unprompted_slice_3d_dsc=float(np.nanmean(per_case_unprompted)) if per_case_unprompted else float("nan"),
                whole_volume_3d_dsc=float(np.nanmean(per_case_whole)) if per_case_whole else float("nan"),
                unprompted_mean_2d_dice=float(np.nanmean(per_case_2d)) if per_case_2d else float("nan"),
            )
        )

    return results, checkpoint_metric(results)
