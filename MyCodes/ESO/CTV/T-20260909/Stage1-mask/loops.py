#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np
import torch

from experiment_core import (
    DiceBCELoss,
    ValidationResult,
    checkpoint_metric,
    dice_3d_from_logits,
    mean_unprompted_2d_dice,
    positive_slice_indices,
    sample_train_prompt_indices,
    unprompted_only_loss,
    unprompted_presence_bce,
    unprompted_slice_3d_dsc,
)

# Reuse the tracking implementation without modifying it.
from bidirectional_tracking import (
    bidirectional_outputs,
    single_direction_outputs,
)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _video_id(batch) -> int:
    return int(batch.metadata.unique_objects_identifier[0, 0, 0].item())


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool = True,
    min_prompts: int = 1,
    max_prompts: int = 5,
    min_prompt_gap: int = 2,
    use_bidirectional_train: bool = False,
    grad_clip_norm: float = 1.0,
    forward_backbone_per_frame: bool = False,
    epoch: int = 0,
    seed: int = 20260909,
    presence_loss_weight: float = 0.05,
) -> dict[str, float]:
    model.train(True)
    criterion = DiceBCELoss().to(device)
    total_loss = 0.0
    total_unprompted_dsc = 0.0
    total_seg_loss = 0.0
    total_presence_loss = 0.0
    n_batch = 0
    core_model = unwrap_model(model)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        video_id = _video_id(batch)
        prompt_rng = random.Random(int(seed) + int(epoch) * 1000003 + video_id)
        positive_indices = positive_slice_indices(batch.masks)
        prompt_frames = sample_train_prompt_indices(
            positive_indices, min_prompts, max_prompts, min_prompt_gap, prompt_rng
        )
        assert 1 <= len(prompt_frames) <= 5
        assert set(prompt_frames).issubset(set(positive_indices))
        assert all(
            b - a >= min_prompt_gap
            for a, b in zip(prompt_frames[:-1], prompt_frames[1:])
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype
        ):
            if use_bidirectional_train:
                outputs = bidirectional_outputs(
                    core_model, batch, prompt_frames, forward_backbone_per_frame
                )
            else:
                direction_rng = random.Random(
                    int(seed) + int(epoch) * 1000003 + video_id + 7919
                )
                outputs = single_direction_outputs(
                    core_model,
                    batch,
                    prompt_frames,
                    reverse=(direction_rng.random() < 0.5),
                    forward_backbone_per_frame=forward_backbone_per_frame,
                )
            seg_loss = unprompted_only_loss(outputs, batch.masks, prompt_frames, criterion)
            if use_bidirectional_train:
                forward_presence_outputs = [
                    {"multistep_object_score_logits": [out["directional_object_score_logits"][0]]}
                    for out in outputs
                ]
                backward_presence_outputs = [
                    {"multistep_object_score_logits": [out["directional_object_score_logits"][1]]}
                    for out in outputs
                ]
                presence_forward = unprompted_presence_bce(
                    forward_presence_outputs, batch.masks, prompt_frames
                )
                presence_backward = unprompted_presence_bce(
                    backward_presence_outputs, batch.masks, prompt_frames
                )
                presence_loss = 0.5 * (presence_forward + presence_backward)
            else:
                presence_loss = unprompted_presence_bce(outputs, batch.masks, prompt_frames)
            loss = seg_loss + float(presence_loss_weight) * presence_loss

        if loss is None or not loss.requires_grad:
            continue
        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(grad_clip_norm)
            )
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item())
        total_seg_loss += float(seg_loss.detach().item())
        total_presence_loss += float(presence_loss.detach().item())
        total_unprompted_dsc += unprompted_slice_3d_dsc(
            outputs, batch.masks, prompt_frames
        )
        n_batch += 1

    if n_batch == 0:
        return {
            "loss": 0.0, "seg_loss": 0.0, "presence_loss": 0.0,
            "unprompted_slice_3d_dsc": 0.0,
        }
    return {
        "loss": total_loss / n_batch,
        "seg_loss": total_seg_loss / n_batch,
        "presence_loss": total_presence_loss / n_batch,
        "unprompted_slice_3d_dsc": total_unprompted_dsc / n_batch,
    }


@torch.no_grad()
def validate_fixed_plan(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    validation_plan: Mapping[str, Mapping[str, list[dict]]],
    prompt_ks=range(1, 6),
    forward_backbone_per_frame: bool = False,
    amp_enabled: bool = True,
) -> tuple[list[ValidationResult], float]:
    """Evaluate exactly two immutable placements per patient and K."""
    model.train(False)
    core_model = unwrap_model(model)
    per_k = {int(k): {"un": [], "whole": [], "d2": []} for k in prompt_ks}

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        patient_key = str(_video_id(batch))
        if patient_key not in validation_plan:
            raise KeyError(f"Patient {patient_key} is missing from validation prompt plan")

        positive_set = set(positive_slice_indices(batch.masks))
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype
        ):
            # Validation runs under no_grad(): encode the complete patient exactly
            # once and reuse the same backbone features for all K x placements x directions.
            # Call forward_image directly so this behavior is independent of
            # core_model.forward_backbone_per_frame_for_eval.
            base_backbone_out = core_model.forward_image(batch.flat_img_batch)
        for k in prompt_ks:
            placements = validation_plan[patient_key].get(str(int(k)), [])
            if len(placements) != 2:
                raise ValueError(
                    f"Patient {patient_key}, K={k} must have exactly 2 validation placements; "
                    f"found {len(placements)}"
                )
            case_un, case_whole, case_d2 = [], [], []
            for placement in placements:
                prompt_frames = [int(x) for x in placement["prompt_frame_ids"]]
                if not set(prompt_frames).issubset(positive_set):
                    raise ValueError(
                        f"Patient {patient_key}, K={k} placement contains CTV-negative slices: "
                        f"{sorted(set(prompt_frames) - positive_set)}"
                    )
                with torch.cuda.amp.autocast(
                    enabled=(device.type == "cuda" and amp_enabled), dtype=amp_dtype
                ):
                    outputs = bidirectional_outputs(
                        core_model,
                        batch,
                        prompt_frames,
                        forward_backbone_per_frame=False,
                        base_backbone_out=base_backbone_out,
                    )
                case_un.append(
                    unprompted_slice_3d_dsc(outputs, batch.masks, prompt_frames)
                )
                case_whole.append(dice_3d_from_logits(outputs, batch.masks))
                case_d2.append(
                    mean_unprompted_2d_dice(outputs, batch.masks, prompt_frames)
                )
            # Average the two placements first, then average patients below.
            per_k[int(k)]["un"].append(float(np.nanmean(case_un)))
            per_k[int(k)]["whole"].append(float(np.nanmean(case_whole)))
            per_k[int(k)]["d2"].append(float(np.nanmean(case_d2)))

    results = []
    for k in prompt_ks:
        vals = per_k[int(k)]
        results.append(
            ValidationResult(
                k=int(k),
                unprompted_slice_3d_dsc=float(np.nanmean(vals["un"])),
                whole_volume_3d_dsc=float(np.nanmean(vals["whole"])),
                unprompted_mean_2d_dice=float(np.nanmean(vals["d2"])),
            )
        )
    return results, checkpoint_metric(results)
