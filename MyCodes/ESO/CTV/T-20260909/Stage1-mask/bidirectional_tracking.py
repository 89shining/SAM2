#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from experiment_core import fuse_bidirectional_outputs


def _raw_sam_output_hook(captured: dict):
    """Capture decoder outputs before SAM2 applies its object-presence mask gate."""
    def hook(_module, _inputs, output):
        captured["low_res_multimasks"] = output[0]
        captured["ious"] = output[1]
    return hook


def _expose_raw_segmentation_output(core_model, current_out: dict, captured: dict) -> None:
    """Expose raw logits for segmentation while leaving native gated memory intact."""
    if not captured:
        raise RuntimeError("SAM mask decoder did not produce an output for this frame")
    raw_multimasks = captured["low_res_multimasks"].float()
    ious = captured["ious"]
    if raw_multimasks.shape[1] > 1:
        batch_indices = torch.arange(raw_multimasks.shape[0], device=raw_multimasks.device)
        best_indices = torch.argmax(ious, dim=-1)
        raw_low_res = raw_multimasks[batch_indices, best_indices].unsqueeze(1)
    else:
        raw_low_res = raw_multimasks
    raw_high_res = F.interpolate(
        raw_low_res,
        size=(core_model.image_size, core_model.image_size),
        mode="bilinear",
        align_corners=False,
    )
    current_out["pred_masks"] = raw_low_res
    current_out["pred_masks_high_res"] = raw_high_res


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


def track_in_order(
    core_model,
    base_backbone_out: dict,
    batch,
    prompt_frames: Sequence[int],
    frame_order: Sequence[int],
    track_in_reverse: bool,
    gt_masks_for_track_step: bool = True,
) -> list[dict]:
    prompt_frames = sorted(set(int(x) for x in prompt_frames))
    prompt_set = set(prompt_frames)
    frame_order = [int(x) for x in frame_order]
    backbone_out = build_backbone_prompt_state(base_backbone_out, batch, prompt_frames, frame_order)
    img_feats_already_computed = base_backbone_out["backbone_fpn"] is not None

    if img_feats_already_computed:
        _, vision_feats, vision_pos_embeds, feat_sizes = core_model._prepare_backbone_features(base_backbone_out)

    num_frames = int(backbone_out["num_frames"])
    cond_order = sorted(prompt_frames, reverse=track_in_reverse)
    non_cond_order = [t for t in frame_order if t not in prompt_set]
    processing_order = cond_order + non_cond_order

    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    all_frame_outputs = {}

    for stage_id in processing_order:
        img_ids = batch.flat_obj_to_img_idx[stage_id]
        if img_feats_already_computed:
            current_vision_feats = [x[:, img_ids] for x in vision_feats]
            current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
        else:
            _, current_vision_feats, current_vision_pos_embeds, feat_sizes = core_model._prepare_backbone_features_per_frame(
                batch.flat_img_batch, img_ids
            )

        is_cond = stage_id in prompt_set
        mask_inputs = backbone_out["mask_inputs_per_frame"].get(stage_id) if is_cond else None
        point_inputs = None
        prev_sam_mask_logits = None

        if not is_cond:
            assert mask_inputs is None
            assert point_inputs is None
            assert prev_sam_mask_logits is None
            assert len(output_dict["cond_frame_outputs"]) == len(prompt_frames), (
                "Every mask-conditioning frame must be processed before propagation."
            )

        captured = {}
        hook_handle = None
        if not is_cond:
            hook_handle = core_model.sam_mask_decoder.register_forward_hook(
                _raw_sam_output_hook(captured)
            )
        try:
            current_out = core_model.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=is_cond,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                # Evaluation must expose GT only through mask_inputs on the
                # frozen conditioning slices, never through this auxiliary API.
                gt_masks=(backbone_out["gt_masks_per_frame"].get(stage_id, None)
                          if gt_masks_for_track_step else None),
                frames_to_add_correction_pt=[],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                prev_sam_mask_logits=prev_sam_mask_logits,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if not is_cond:
            # Native hard-gated masks have already been encoded into memory. Replace
            # only unprompted segmentation outputs with continuous raw logits.
            _expose_raw_segmentation_output(core_model, current_out, captured)

        if is_cond:
            output_dict["cond_frame_outputs"][stage_id] = current_out
        else:
            output_dict["non_cond_frame_outputs"][stage_id] = current_out

        all_frame_outputs[stage_id] = current_out

    outputs = [all_frame_outputs[t] for t in range(num_frames)]
    return [{k: v for k, v in out.items() if k != "obj_ptr"} for out in outputs]


def bidirectional_outputs(
    core_model,
    batch,
    prompt_frames: Sequence[int],
    forward_backbone_per_frame: bool = False,
    base_backbone_out: dict | None = None,
    gt_masks_for_track_step: bool = True,
) -> list[dict]:
    if base_backbone_out is None:
        base_backbone_out = precompute_backbone_out(core_model, batch, forward_backbone_per_frame)
    num_frames = int(batch.num_frames)
    outputs_forward = track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=list(range(num_frames)),
        track_in_reverse=False,
        gt_masks_for_track_step=gt_masks_for_track_step,
    )
    outputs_backward = track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=list(range(num_frames - 1, -1, -1)),
        track_in_reverse=True,
        gt_masks_for_track_step=gt_masks_for_track_step,
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
