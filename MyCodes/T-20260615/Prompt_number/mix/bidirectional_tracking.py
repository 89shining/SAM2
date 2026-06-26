#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Sequence

import torch

from experiment_core import fuse_bidirectional_outputs


def precompute_backbone_out(core_model, batch, forward_backbone_per_frame: bool = False) -> dict:
    if forward_backbone_per_frame:
        return {"backbone_fpn": None, "vision_pos_enc": None}
    if core_model.training or not core_model.forward_backbone_per_frame_for_eval:
        return core_model.forward_image(batch.flat_img_batch)
    return {"backbone_fpn": None, "vision_pos_enc": None}


def _clone_backbone_out(base_backbone_out: dict) -> dict:
    out = {}
    for k, v in base_backbone_out.items():
        out[k] = v
    return out


def build_backbone_prompt_state(base_backbone_out: dict, batch, prompt_frames: Sequence[int], frame_order: Sequence[int]) -> dict:
    prompt_frames = list(dict.fromkeys(int(x) for x in prompt_frames))
    frame_order = [int(x) for x in frame_order]
    prompt_set = set(prompt_frames)

    backbone_out = _clone_backbone_out(base_backbone_out)
    gt_masks_per_frame = {frame_idx: masks.unsqueeze(1) for frame_idx, masks in enumerate(batch.masks)}
    backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
    backbone_out["num_frames"] = int(batch.num_frames)
    backbone_out["use_pt_input"] = False
    backbone_out["point_inputs_per_frame"] = {}
    backbone_out["frames_to_add_correction_pt"] = []
    backbone_out["init_cond_frames"] = prompt_frames
    backbone_out["frames_not_in_init_cond"] = [t for t in frame_order if t not in prompt_set]
    backbone_out["mask_inputs_per_frame"] = {t: gt_masks_per_frame[t] for t in prompt_frames}
    return backbone_out


def _empty_point_prompt(num_obj: int, device: torch.device):
    return {
        "point_coords": torch.zeros((num_obj, 0, 2), dtype=torch.float32, device=device),
        "point_labels": torch.zeros((num_obj, 0), dtype=torch.int32, device=device),
    }


def track_in_order(
    core_model,
    base_backbone_out: dict,
    batch,
    prompt_frames: Sequence[int],
    frame_order: Sequence[int],
    track_in_reverse: bool,
) -> list[dict]:
    frame_order = [int(x) for x in frame_order]
    backbone_out = build_backbone_prompt_state(base_backbone_out, batch, prompt_frames, frame_order)
    img_feats_already_computed = base_backbone_out["backbone_fpn"] is not None

    if img_feats_already_computed:
        _, vision_feats, vision_pos_embeds, feat_sizes = core_model._prepare_backbone_features(base_backbone_out)

    num_frames = int(backbone_out["num_frames"])
    init_cond_set = set(backbone_out["init_cond_frames"])
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    all_frame_outputs = {}
    prev_mask_logits_per_frame = {}

    for order_idx, stage_id in enumerate(frame_order):
        img_ids = batch.flat_obj_to_img_idx[stage_id]
        if img_feats_already_computed:
            current_vision_feats = [x[:, img_ids] for x in vision_feats]
            current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
        else:
            _, current_vision_feats, current_vision_pos_embeds, feat_sizes = core_model._prepare_backbone_features_per_frame(
                batch.flat_img_batch, img_ids
            )

        mask_inputs = backbone_out["mask_inputs_per_frame"].get(stage_id, None)
        point_inputs = backbone_out["point_inputs_per_frame"].get(stage_id, None)
        prev_logits = None
        if stage_id not in init_cond_set and mask_inputs is None and order_idx > 0:
            prev_frame_id = frame_order[order_idx - 1]
            prev_logits = prev_mask_logits_per_frame.get(prev_frame_id, None)
            if prev_logits is not None and point_inputs is None:
                point_inputs = _empty_point_prompt(prev_logits.shape[0], prev_logits.device)

        current_out = core_model.track_step(
            frame_idx=stage_id,
            is_init_cond_frame=stage_id in init_cond_set,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
            frames_to_add_correction_pt=[],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            prev_sam_mask_logits=prev_logits,
        )

        if stage_id in init_cond_set:
            output_dict["cond_frame_outputs"][stage_id] = current_out
            output_dict["non_cond_frame_outputs"].pop(stage_id, None)
        else:
            output_dict["non_cond_frame_outputs"][stage_id] = current_out
            output_dict["cond_frame_outputs"].pop(stage_id, None)

        all_frame_outputs[stage_id] = current_out
        prev_mask_logits_per_frame[stage_id] = torch.clamp(current_out["pred_masks"].detach(), -32.0, 32.0)

    outputs = [all_frame_outputs[t] for t in range(num_frames)]
    return [{k: v for k, v in out.items() if k != "obj_ptr"} for out in outputs]


def bidirectional_outputs(
    core_model,
    batch,
    prompt_frames: Sequence[int],
    forward_backbone_per_frame: bool = False,
) -> list[dict]:
    base_backbone_out = precompute_backbone_out(core_model, batch, forward_backbone_per_frame)
    num_frames = int(batch.num_frames)
    outputs_forward = track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=list(range(num_frames)),
        track_in_reverse=False,
    )
    outputs_backward = track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=list(range(num_frames - 1, -1, -1)),
        track_in_reverse=True,
    )
    return fuse_bidirectional_outputs(outputs_forward, outputs_backward)


def single_direction_outputs(
    core_model,
    batch,
    prompt_frames: Sequence[int],
    reverse: bool,
    forward_backbone_per_frame: bool = False,
) -> list[dict]:
    base_backbone_out = precompute_backbone_out(core_model, batch, forward_backbone_per_frame)
    num_frames = int(batch.num_frames)
    frame_order = list(range(num_frames - 1, -1, -1)) if reverse else list(range(num_frames))
    return track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=frame_order,
        track_in_reverse=reverse,
    )
