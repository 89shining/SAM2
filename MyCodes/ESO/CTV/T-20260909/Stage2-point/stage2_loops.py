"""Patient-level Stage-2 point-correction train/validation loops."""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from point_clicker import CorrectionPoint, sample_correction_point
from stage1_bridge import DiceBCELoss, positive_slice_indices, sample_train_prompt_indices
from stage2_tracking import bidirectional_mixed_outputs, hard_prediction, stacked_logits


def _video_id(batch) -> int:
    return int(batch.metadata.unique_objects_identifier[0, 0, 0].item())


def _gt_volume_zyx(batch) -> torch.Tensor:
    """Return the single CTV object as a canonical [Z,Y,X] tensor."""
    masks = batch.masks
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(
            f"Stage2 requires one object with masks [Z,1,Y,X], got {tuple(masks.shape)}"
        )
    return masks[:, 0]


def _initial_outputs(model, batch, prompt_frames: Sequence[int], base_backbone=None, trace=None):
    """Replay only the clinician-supplied initial mask prompts."""
    placeholder_prior = torch.zeros(
        (int(batch.num_frames), *batch.masks.shape[-2:]),
        device=batch.masks.device,
        dtype=torch.bool,
    )
    return bidirectional_mixed_outputs(
        model, batch, prompt_frames, placeholder_prior, [], base_backbone, trace
    )


def _hard_mask_initialization(
    model, batch, prompt_frames: Sequence[int], base_backbone=None, trace=None
) -> torch.Tensor:
    """Current Stage-2 model's mask-guided P0, returned as [Z,Y,X] bool."""
    with torch.inference_mode():
        if base_backbone is None:
            base_backbone = model.forward_image(batch.flat_img_batch)
        outputs = _initial_outputs(model, batch, prompt_frames, base_backbone, trace)
        return hard_prediction(outputs)


def _terminal_loss(
    criterion: DiceBCELoss,
    outputs,
    gt_zyx: torch.Tensor,
    initial_mask_frames: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = stacked_logits(outputs)[:, 0]
    keep = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    keep[list(map(int, initial_mask_frames))] = False
    if not bool(keep.any()):
        raise RuntimeError("No non-initial-mask slices remain for Stage2 terminal loss")
    segmentation = criterion(logits[keep], gt_zyx[keep].float())
    presence_target = gt_zyx.reshape(gt_zyx.shape[0], -1).any(dim=1).float()
    forward_presence = torch.stack(
        [out["stage2_forward_object_score_logits"].reshape(-1)[0] for out in outputs]
    )
    reverse_presence = torch.stack(
        [out["stage2_reverse_object_score_logits"].reshape(-1)[0] for out in outputs]
    )
    presence = 0.5 * (
        F.binary_cross_entropy_with_logits(forward_presence[keep].float(), presence_target[keep])
        + F.binary_cross_entropy_with_logits(reverse_presence[keep].float(), presence_target[keep])
    )
    return segmentation + 0.05 * presence, logits, segmentation, presence


def _whole_volume_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred, gt = pred.bool(), gt.bool()
    denom = pred.sum() + gt.sum()
    if int(denom) == 0:
        return 1.0
    return float(((2 * (pred & gt).sum().float() + 1e-6) / (denom.float() + 1e-6)).item())


def _point_row(point: CorrectionPoint, round_index: int, dice_before: float, dice_after: float | None) -> dict:
    return {
        "round": int(round_index), "z": point.z, "y": point.y, "x": point.x,
        "label": point.label, "error_component_type": point.error_type,
        "error_component_voxels": point.component_voxels,
        "error_component_mm3": point.component_mm3,
        "dice_before": dice_before, "dice_after": "" if dice_after is None else dice_after,
    }


def train_patient_episode(
    model,
    batch,
    spacing_zyx: Sequence[float],
    rng: random.Random,
    min_prompt_gap: int = 2,
) -> tuple[torch.Tensor | None, dict]:
    """Build a no-grad trajectory and return loss only for its terminal state."""
    gt = _gt_volume_zyx(batch).bool()
    gt_numpy = gt.detach().cpu().numpy()
    prompt_frames = sample_train_prompt_indices(
        positive_slice_indices(gt), 1, 5, min_prompt_gap, rng
    )
    terminal_round = rng.randint(0, 5)
    clicks: list[CorrectionPoint] = []
    logs: list[dict] = []
    criterion = DiceBCELoss().to(gt.device)

    if terminal_round == 0:
        # T=0 is a genuinely trainable initialization episode. Per-frame
        # backbone evaluation preserves Image Encoder LoRA gradients without
        # retaining a full-volume encoder graph.
        initial_outputs = _initial_outputs(model, batch, prompt_frames)
        loss, logits, loss_seg, loss_presence = _terminal_loss(
            criterion, initial_outputs, gt, prompt_frames
        )
        terminal_hard = logits.detach().gt(0.0)
        return loss, {
            "initial_prompt_frames": list(map(int, prompt_frames)),
            "sampled_T": 0, "effective_T": 0, "clicks": [],
            "terminal_dice": _whole_volume_dice(terminal_hard, gt),
            "loss_seg": float(loss_seg.detach()),
            "loss_presence": float(loss_presence.detach()),
        }

    # Cache one detached Stage-2 image encoding for all intermediate no-grad
    # trajectory rounds. The terminal differentiable round uses per-frame
    # backbone evaluation to avoid retaining a full-volume Image Encoder graph.
    with torch.no_grad():
        trajectory_backbone = model.forward_image(batch.flat_img_batch)
    previous = _hard_mask_initialization(model, batch, prompt_frames, trajectory_backbone)
    terminal_outputs = None
    for round_index in range(1, terminal_round + 1):
        click = sample_correction_point(
            gt_numpy, previous.detach().cpu().numpy(), spacing_zyx, "train", rng,
            exclude_slices=prompt_frames,
        )
        if click is None:
            # Exact P(t-1)=GT: no fabricated click and no trainable terminal
            # state exists. This is correctly a zero-update episode.
            break
        clicks.append(click)
        before = _whole_volume_dice(previous, gt)
        if round_index < terminal_round:
            with torch.no_grad():
                intermediate = bidirectional_mixed_outputs(
                    model, batch, prompt_frames, previous, clicks, trajectory_backbone
                )
                next_hard = hard_prediction(intermediate)
            logs.append(_point_row(click, round_index, before, _whole_volume_dice(next_hard, gt)))
            if torch.equal(next_hard, gt):
                # The no-grad trajectory reached an exact mask before sampled T.
                # Re-evaluate this same state once with gradients, using the same
                # prior and accumulated clicks, so the effective terminal state
                # still has the protocol-defined full-volume terminal loss.
                terminal_outputs = bidirectional_mixed_outputs(
                    model, batch, prompt_frames, previous, clicks, base_backbone_out=None
                )
                loss, logits, loss_seg, loss_presence = _terminal_loss(
                    criterion, terminal_outputs, gt, prompt_frames
                )
                terminal_hard = logits.detach().gt(0.0)
                logs[-1]["dice_after"] = _whole_volume_dice(terminal_hard, gt)
                return loss, {
                    "initial_prompt_frames": list(map(int, prompt_frames)),
                    "sampled_T": int(terminal_round), "effective_T": int(round_index),
                    "clicks": logs, "terminal_dice": _whole_volume_dice(terminal_hard, gt),
                    "loss_seg": float(loss_seg.detach()),
                    "loss_presence": float(loss_presence.detach()),
                    "early_perfect": True,
                }
            previous = next_hard
            continue
        terminal_outputs = bidirectional_mixed_outputs(
            model, batch, prompt_frames, previous, clicks, base_backbone_out=None
        )
        loss, logits, loss_seg, loss_presence = _terminal_loss(
            criterion, terminal_outputs, gt, prompt_frames
        )
        terminal_hard = logits.detach().gt(0.0)
        logs.append(_point_row(click, round_index, before, _whole_volume_dice(terminal_hard, gt)))
        return loss, {
            "initial_prompt_frames": list(map(int, prompt_frames)),
            "sampled_T": int(terminal_round), "effective_T": int(round_index),
            "clicks": logs, "terminal_dice": _whole_volume_dice(terminal_hard, gt),
            "loss_seg": float(loss_seg.detach()),
            "loss_presence": float(loss_presence.detach()),
        }
    return None, {
        "initial_prompt_frames": list(map(int, prompt_frames)),
        "sampled_T": int(terminal_round), "effective_T": len(clicks),
        "clicks": logs, "terminal_dice": _whole_volume_dice(previous, gt),
        "perfect_before_trainable_terminal": True,
    }


@torch.no_grad()
def validate_k3_t5(
    model,
    loader,
    validation_plan: Mapping[str, Mapping],
    spacing_by_patient: Mapping[int, Sequence[float]],
) -> tuple[float, list[dict], dict]:
    """Fixed K=3, two placements, P0--P5 Dice and workflow mean."""
    model.eval()
    patient_rows: list[dict] = []
    for batch in loader:
        batch = batch.to(next(model.parameters()).device, non_blocking=True)
        patient = _video_id(batch)
        record = validation_plan[str(patient)]
        placements = record["placements"]["3"]
        if len(placements) != 2:
            raise ValueError(f"Patient {patient}: Stage2 validation requires exactly two K=3 placements")
        placement_curves = []
        backbone = model.forward_image(batch.flat_img_batch)
        for placement in placements:
            prompts = [int(value) for value in placement["prompt_frame_ids"]]
            previous = _hard_mask_initialization(model, batch, prompts, backbone)
            clicks: list[CorrectionPoint] = []
            curve = [_whole_volume_dice(previous, _gt_volume_zyx(batch))]
            for _round in range(1, 6):
                click = sample_correction_point(
                    _gt_volume_zyx(batch).detach().cpu().numpy(), previous.detach().cpu().numpy(),
                    spacing_by_patient[patient], "validation",
                    exclude_slices=prompts,
                )
                if click is None:
                    curve.append(curve[-1])
                    continue
                clicks.append(click)
                outputs = bidirectional_mixed_outputs(
                    model, batch, prompts, previous, clicks, backbone
                )
                previous = hard_prediction(outputs)
                curve.append(_whole_volume_dice(previous, _gt_volume_zyx(batch)))
            placement_curves.append(curve)
        averaged = np.asarray(placement_curves, dtype=np.float64).mean(axis=0)
        row = {"patient_id": patient}
        for t in range(6):
            row[f"placement0_dice_p{t}"] = placement_curves[0][t]
            row[f"placement1_dice_p{t}"] = placement_curves[1][t]
            row[f"dice_p{t}"] = float(averaged[t])
        row["workflow_score_p0_p5"] = float(averaged.mean())
        row["gain_p5_minus_p0"] = float(averaged[5] - averaged[0])
        patient_rows.append(row)
    mean_curve = {
        f"D{t}": float(np.mean([row[f"dice_p{t}"] for row in patient_rows]))
        for t in range(6)
    }
    score = float(np.mean([mean_curve[f"D{t}"] for t in range(6)]))
    summary = {
        **mean_curve,
        "workflow_score_p0_p5": score,
        "gain_D5_minus_D0": mean_curve["D5"] - mean_curve["D0"],
    }
    return score, patient_rows, summary
