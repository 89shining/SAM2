"""Mixed mask-initialization and point-correction propagation for SAM2 Stage 2."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from point_clicker import CorrectionPoint


def _capture_decoder(captured: dict):
    def hook(_module, _inputs, output):
        captured["multimasks"] = output[0]
        captured["ious"] = output[1]
    return hook


def _replace_with_raw_logits(core_model, output: dict, captured: dict) -> None:
    """Keep native gated memory, but expose continuous decoder logits for loss."""
    if "multimasks" not in captured:
        raise RuntimeError("SAM decoder output was not captured")
    multimasks, ious = captured["multimasks"].float(), captured["ious"]
    if multimasks.shape[1] > 1:
        idx = torch.argmax(ious, dim=-1)
        batch_idx = torch.arange(multimasks.shape[0], device=multimasks.device)
        low = multimasks[batch_idx, idx].unsqueeze(1)
    else:
        low = multimasks
    output["pred_masks"] = low
    output["pred_masks_high_res"] = F.interpolate(
        low, size=(core_model.image_size, core_model.image_size),
        mode="bilinear", align_corners=False,
    )


def _point_inputs(points: Iterable[CorrectionPoint], device: torch.device) -> dict[int, dict]:
    grouped: dict[int, list[CorrectionPoint]] = defaultdict(list)
    for point in points:
        grouped[int(point.z)].append(point)
    result = {}
    for z, items in grouped.items():
        # SAM2 uses (x, y) and binary labels: foreground=1, background=0.
        coords = torch.tensor([[[p.x, p.y] for p in items]], dtype=torch.float32, device=device)
        labels = torch.tensor([[1 if p.label > 0 else 0 for p in items]], dtype=torch.int64, device=device)
        result[z] = {"point_coords": coords, "point_labels": labels}
    return result


def _prior_logits(previous_hard_tyx: torch.Tensor, frame: int) -> torch.Tensor:
    """Convert the previous hard prediction on a conditioning frame to a SAM mask prior."""
    prior = previous_hard_tyx[int(frame)].to(dtype=torch.float32)
    return (prior.mul(20.0).sub(10.0)).unsqueeze(0).unsqueeze(0)


def _prepare_features(core_model, base_backbone_out: dict):
    if base_backbone_out.get("backbone_fpn") is None:
        raise RuntimeError("Invalid cached full-volume backbone features")
    _, vision_feats, vision_pos, feat_sizes = core_model._prepare_backbone_features(base_backbone_out)
    return vision_feats, vision_pos, feat_sizes


def _track_direction(
    core_model,
    batch,
    initial_mask_frames: Sequence[int],
    previous_hard_tyx: torch.Tensor,
    points: Sequence[CorrectionPoint],
    reverse: bool,
    base_backbone_out: dict | None,
    trace: list[dict] | None = None,
) -> list[dict]:
    cached = base_backbone_out is not None
    if cached:
        vision_feats, vision_pos, feat_sizes = _prepare_features(core_model, base_backbone_out)
    point_by_frame = _point_inputs(points, previous_hard_tyx.device)
    mask_set = {int(frame) for frame in initial_mask_frames}
    point_set = set(point_by_frame)
    overlap = mask_set & point_set
    if overlap:
        raise RuntimeError(f"Correction points cannot lie on initial-mask slices: {sorted(overlap)}")
    cond_frames = sorted(mask_set | point_set, reverse=reverse)
    if not cond_frames:
        raise ValueError("Stage2 mixed propagation needs at least one conditioning frame")
    cond_set = set(cond_frames)
    frame_order = list(range(int(batch.num_frames) - 1, -1, -1)) if reverse else list(range(int(batch.num_frames)))
    processing = cond_frames + [frame for frame in frame_order if frame not in cond_set]
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    all_outputs = {}
    for frame in processing:
        img_ids = batch.flat_obj_to_img_idx[frame]
        if cached:
            current_feats = [feature[:, img_ids] for feature in vision_feats]
            current_pos = [position[:, img_ids] for position in vision_pos]
        else:
            _, current_feats, current_pos, feat_sizes = core_model._prepare_backbone_features_per_frame(
                batch.flat_img_batch, img_ids
            )
        is_cond = frame in cond_set
        is_mask_cond = frame in mask_set
        is_point_cond = frame in point_set
        # Construct once so the trace reports the exact objects passed into
        # track_step rather than a second, inferred description of the state.
        point_arg = point_by_frame.get(frame) if is_point_cond else None
        mask_arg = batch.masks[frame].unsqueeze(1) if is_mask_cond else None
        gt_arg = batch.masks[frame].unsqueeze(1) if is_mask_cond else None
        prior_arg = _prior_logits(previous_hard_tyx, frame) if is_point_cond else None
        captured = {}
        hook = None
        # A supplied GT mask frame must retain SAM2's native _use_mask_as_output
        # result. Raw decoder logits are exposed on point and propagated frames.
        if not is_mask_cond:
            hook = core_model.sam_mask_decoder.register_forward_hook(_capture_decoder(captured))
        try:
            current = core_model.track_step(
                frame_idx=frame,
                is_init_cond_frame=is_cond,
                current_vision_feats=current_feats,
                current_vision_pos_embeds=current_pos,
                feat_sizes=feat_sizes,
                point_inputs=point_arg,
                mask_inputs=mask_arg,
                # GT is available to SAM2 only on the clinician-supplied initial
                # mask frames. Point and propagated frames must not receive it.
                gt_masks=gt_arg,
                frames_to_add_correction_pt=[],
                output_dict=output_dict,
                num_frames=int(batch.num_frames),
                track_in_reverse=reverse,
                prev_sam_mask_logits=prior_arg,
            )
        finally:
            if hook is not None:
                hook.remove()
        # Memory was already built from the native output. Expose continuous raw
        # logits only for point-conditioned and propagated frames; initial GT-mask
        # frames retain SAM2's native supplied-mask output.
        if not is_mask_cond:
            _replace_with_raw_logits(core_model, current, captured)
        if trace is not None:
            trace.append({
                "direction": "reverse" if reverse else "forward",
                "frame": int(frame),
                "processing_index": len(trace),
                "is_initial_mask_frame": bool(is_mask_cond),
                "is_point_frame": bool(is_point_cond),
                "mask_inputs": mask_arg is not None,
                "point_inputs": point_arg is not None,
                "gt_masks": gt_arg is not None,
                "prev_sam_mask_logits_shape": (
                    None if prior_arg is None else list(prior_arg.shape)
                ),
            })
        (output_dict["cond_frame_outputs"] if is_cond else output_dict["non_cond_frame_outputs"])[frame] = current
        all_outputs[frame] = current
    return [{key: value for key, value in all_outputs[frame].items() if key != "obj_ptr"} for frame in range(int(batch.num_frames))]


def bidirectional_mixed_outputs(
    core_model,
    batch,
    initial_mask_frames: Sequence[int],
    previous_hard_tyx: torch.Tensor,
    points: Sequence[CorrectionPoint],
    base_backbone_out: dict | None = None,
    trace: list[dict] | None = None,
) -> list[dict]:
    """Replay initial mask prompts and all accumulated point corrections."""
    forward = _track_direction(
        core_model, batch, initial_mask_frames, previous_hard_tyx,
        points, False, base_backbone_out, trace,
    )
    backward = _track_direction(
        core_model, batch, initial_mask_frames, previous_hard_tyx,
        points, True, base_backbone_out, trace,
    )
    fused = []
    for out_f, out_b in zip(forward, backward):
        probability = 0.5 * (torch.sigmoid(out_f["pred_masks_high_res"]) + torch.sigmoid(out_b["pred_masks_high_res"]))
        out = dict(out_f)
        out["pred_masks_high_res"] = torch.logit(probability.clamp(1e-4, 1.0 - 1e-4))
        # Segmentation is probability-fused, while presence supervision remains
        # direction-specific so both native memory pathways receive gradients.
        out["stage2_forward_object_score_logits"] = out_f["multistep_object_score_logits"][-1]
        out["stage2_reverse_object_score_logits"] = out_b["multistep_object_score_logits"][-1]
        fused.append(out)
    return fused


def stacked_logits(outputs: Sequence[dict]) -> torch.Tensor:
    return torch.stack([output["pred_masks_high_res"][:, 0] for output in outputs], dim=0)


def hard_prediction(outputs: Sequence[dict]) -> torch.Tensor:
    logits = stacked_logits(outputs)
    if logits.shape[1] != 1:
        raise ValueError(f"Stage2 currently requires patient-level batch size 1, got {tuple(logits.shape)}")
    return logits[:, 0].detach().gt(0.0)
