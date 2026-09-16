"""Patient-level native-state Stage-2 train and validation loops."""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from point_clicker import CorrectionPoint, sample_correction_point
from stage1_bridge import DiceBCELoss, positive_slice_indices, sample_train_prompt_indices
from stage2_tracking import OfficialMultiFrameBidirectionalState, hard_prediction, stacked_logits


def _video_id(batch) -> int:
    return int(batch.metadata.unique_objects_identifier[0, 0, 0].item())


def _gt_volume_zyx(batch) -> torch.Tensor:
    masks = batch.masks
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"Stage2 requires one object with masks [Z,1,Y,X], got {tuple(masks.shape)}")
    return masks[:, 0]


def _new_state(model, batch, prompt_frames, base_backbone=None, trace=None):
    return OfficialMultiFrameBidirectionalState(
        model, batch, prompt_frames, base_backbone, trace
    )


def _initial_outputs(model, batch, prompt_frames: Sequence[int], base_backbone=None, trace=None):
    """Build P0 after registering each initial GT mask exactly once."""
    return _new_state(model, batch, prompt_frames, base_backbone, trace).outputs()


def _hard_mask_initialization(
    model, batch, prompt_frames: Sequence[int], base_backbone=None, trace=None
) -> torch.Tensor:
    with torch.no_grad():
        return hard_prediction(_initial_outputs(model, batch, prompt_frames, base_backbone, trace))


def _terminal_loss(
    criterion: DiceBCELoss, outputs, gt_zyx: torch.Tensor, initial_mask_frames: Sequence[int]
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


def _terminal_replay(model, batch, prompt_frames, clicks):
    """Use only for T=0, where P0 itself is the differentiable terminal state."""
    terminal = _new_state(model, batch, prompt_frames)
    for click in clicks:
        terminal.add_click(click)
    return terminal.outputs()


def terminal_transition_from_history(model, batch, prompt_frames, clicks):
    """Differentiate only the final real click after rebuilding S(t-1).

    For an early-perfect trajectory, there is no valid propagation-only
    transition in the native multi-frame state.  Rebuild the state immediately
    before the last actual click under ``no_grad`` and replay that click with
    autograd instead of invoking the legacy memory-decoupled API.
    """
    if not clicks:
        return _terminal_replay(model, batch, prompt_frames, [])
    with torch.no_grad():
        history_backbone = model.forward_image(batch.flat_img_batch)
        history = _new_state(model, batch, prompt_frames, history_backbone)
        for historical_click in clicks[:-1]:
            history.add_click(historical_click)
    terminal = history.detached_terminal_snapshot()
    terminal.add_click(clicks[-1])
    return terminal.outputs()


def train_patient_episode(
    model, batch, spacing_zyx: Sequence[float], rng: random.Random, min_prompt_gap: int = 2
) -> tuple[torch.Tensor | None, dict]:
    """Train T=0 fully; for T>=1 backpropagate only through terminal transition."""
    gt = _gt_volume_zyx(batch).bool()
    gt_numpy = gt.detach().cpu().numpy()
    prompt_frames = sample_train_prompt_indices(
        positive_slice_indices(gt), 1, 5, min_prompt_gap, rng
    )
    terminal_round = rng.randint(0, 5)
    criterion = DiceBCELoss().to(gt.device)
    clicks: list[CorrectionPoint] = []
    logs: list[dict] = []

    if terminal_round == 0:
        outputs = _terminal_replay(model, batch, prompt_frames, [])
        loss, logits, loss_seg, loss_presence = _terminal_loss(criterion, outputs, gt, prompt_frames)
        hard = hard_prediction(outputs)
        return loss, {
            "initial_prompt_frames": list(map(int, prompt_frames)), "sampled_T": 0,
            "effective_T": 0, "clicks": [], "terminal_dice": _whole_volume_dice(hard, gt),
            "loss_seg": float(loss_seg.detach()), "loss_presence": float(loss_presence.detach()),
        }

    # Native trajectory state persists across all historical clicks under
    # no_grad.  The final click is handled below from a detached S(t-1)
    # snapshot, avoiding full BPTT across C1...C(t-1).
    with torch.no_grad():
        trajectory_backbone = model.forward_image(batch.flat_img_batch)
        trajectory = _new_state(model, batch, prompt_frames, trajectory_backbone)
        previous = hard_prediction(trajectory.outputs())

    for round_index in range(1, terminal_round + 1):
        click = sample_correction_point(
            gt_numpy, previous.detach().cpu().numpy(), spacing_zyx, "train", rng,
            exclude_slices=prompt_frames,
        )
        if click is None:
            break
        clicks.append(click)
        before = _whole_volume_dice(previous, gt)
        if round_index < terminal_round:
            with torch.no_grad():
                trajectory.add_click(click)
                next_hard = hard_prediction(trajectory.outputs())
            logs.append(_point_row(click, round_index, before, _whole_volume_dice(next_hard, gt)))
            previous = next_hard
            if torch.equal(previous, gt):
                # Do not invent a propagation-only terminal transition. Replay
                # the final real click from the preceding detached native state.
                outputs = terminal_transition_from_history(model, batch, prompt_frames, clicks)
                loss, logits, loss_seg, loss_presence = _terminal_loss(criterion, outputs, gt, prompt_frames)
                logs[-1]["dice_after"] = _whole_volume_dice(hard_prediction(outputs), gt)
                return loss, {
                    "initial_prompt_frames": list(map(int, prompt_frames)),
                    "sampled_T": int(terminal_round), "effective_T": int(round_index),
                    "clicks": logs, "terminal_dice": _whole_volume_dice(hard_prediction(outputs), gt),
                    "loss_seg": float(loss_seg.detach()), "loss_presence": float(loss_presence.detach()),
                    "early_perfect": True,
                }
            continue

        # Truncated interaction BPTT: clone normal detached S(t-1), then make
        # only Ct, its memory update, and the following propagation trainable.
        terminal = trajectory.detached_terminal_snapshot()
        terminal.add_click(click)
        outputs = terminal.outputs()
        loss, logits, loss_seg, loss_presence = _terminal_loss(criterion, outputs, gt, prompt_frames)
        hard = hard_prediction(outputs)
        logs.append(_point_row(click, round_index, before, _whole_volume_dice(hard, gt)))
        return loss, {
            "initial_prompt_frames": list(map(int, prompt_frames)),
            "sampled_T": int(terminal_round), "effective_T": int(round_index), "clicks": logs,
            "terminal_dice": _whole_volume_dice(hard, gt),
            "loss_seg": float(loss_seg.detach()), "loss_presence": float(loss_presence.detach()),
        }

    return None, {
        "initial_prompt_frames": list(map(int, prompt_frames)),
        "sampled_T": int(terminal_round), "effective_T": len(clicks), "clicks": logs,
        "terminal_dice": _whole_volume_dice(previous, gt),
        "perfect_before_trainable_terminal": True,
    }


@torch.no_grad()
def validate_k3_t5(
    model, loader, validation_plan: Mapping[str, Mapping], spacing_by_patient: Mapping[int, Sequence[float]]
) -> tuple[float, list[dict], dict]:
    """Fixed K=3 native state transitions, P0 through P5."""
    model.eval()
    rows: list[dict] = []
    for batch in loader:
        batch = batch.to(next(model.parameters()).device, non_blocking=True)
        patient = _video_id(batch)
        placements = validation_plan[str(patient)]["placements"]["3"]
        if len(placements) != 2:
            raise ValueError(f"Patient {patient}: validation needs exactly two K=3 placements")
        curves = []
        backbone = model.forward_image(batch.flat_img_batch)
        gt = _gt_volume_zyx(batch)
        gt_numpy = gt.detach().cpu().numpy()
        for placement in placements:
            prompts = [int(value) for value in placement["prompt_frame_ids"]]
            state = _new_state(model, batch, prompts, backbone)
            previous = hard_prediction(state.outputs())
            curve = [_whole_volume_dice(previous, gt)]
            for _ in range(5):
                click = sample_correction_point(
                    gt_numpy, previous.detach().cpu().numpy(), spacing_by_patient[patient],
                    "validation", exclude_slices=prompts,
                )
                if click is not None:
                    state.add_click(click)
                    previous = hard_prediction(state.outputs())
                curve.append(_whole_volume_dice(previous, gt))
            curves.append(curve)
        averaged = np.asarray(curves, dtype=np.float64).mean(axis=0)
        row = {"patient_id": patient}
        for t in range(6):
            row[f"placement0_dice_p{t}"] = curves[0][t]
            row[f"placement1_dice_p{t}"] = curves[1][t]
            row[f"dice_p{t}"] = float(averaged[t])
        row["workflow_score_p0_p5"] = float(averaged.mean())
        row["gain_p5_minus_p0"] = float(averaged[5] - averaged[0])
        rows.append(row)
    mean_curve = {f"D{t}": float(np.mean([row[f"dice_p{t}"] for row in rows])) for t in range(6)}
    score = float(np.mean(list(mean_curve.values())))
    return score, rows, {
        **mean_curve, "workflow_score_p0_p5": score,
        "gain_D5_minus_D0": mean_curve["D5"] - mean_curve["D0"],
    }
