#!/usr/bin/env python
"""Fixed-case, read-only D0--D5 pilot for a Stage-2 checkpoint.

The script uses the locked K=3/two-placement validation plan.  It records
native and raw-decoder Dice separately whenever the tracking implementation
exposes both views; legacy full-history code has one shared decoder-logit view,
which is explicitly marked in the output.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from point_clicker import sample_correction_point
from stage1_bridge import (
    DEFAULT_DATA_ROOT, DEFAULT_INIT_CKPT, DEFAULT_MODEL_CFG, DEFAULT_SPLIT_PATH,
    RectalCTVVolumeDataset, build_model, make_or_load_splits,
)
from training.utils.data_utils import collate_fn


def collate_one(items):
    return collate_fn(items, dict_key="eso_ctv_d0_d5_pilot")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--stage1-ckpt", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--validation-plan", type=Path, required=True)
    p.add_argument("--patient-count", type=int, default=5)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "train")
    p.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    p.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _model_args(cli):
    return SimpleNamespace(
        model_cfg=cli.model_cfg, init_ckpt=cli.init_ckpt, input_size=512,
        lora_r=4, lora_alpha=16, lora_dropout=0.1,
        image_encoder_activation_checkpointing=True,
        fold=cli.fold, stage1_ckpt=cli.stage1_ckpt,
    )


def _dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = pred.bool(), target.bool()
    denom = pred.sum() + target.sum()
    if not int(denom):
        return 1.0
    return float(((2 * (pred & target).sum().float() + 1e-6) / (denom.float() + 1e-6)).item())


def _spacing(patient_dir: Path) -> tuple[float, float, float]:
    sx, sy, sz = sitk.ReadImage(str(patient_dir / "image.nii.gz")).GetSpacing()
    return float(sz), float(sy), float(sx)


def _gt(batch) -> torch.Tensor:
    if batch.masks.ndim != 4 or batch.masks.shape[1] != 1:
        raise ValueError(f"Expected [Z,1,Y,X], got {tuple(batch.masks.shape)}")
    return batch.masks[:, 0].bool()


def _logits(outputs, field: str) -> torch.Tensor:
    # Full-history baseline exposes only its decoder-logit view, so raw falls
    # back to that same field and the report makes this explicit.
    field = field if field in outputs[0] else "pred_masks_high_res"
    values = torch.stack([out[field][:, 0] for out in outputs], dim=0)
    if values.shape[1] != 1:
        raise ValueError(f"Pilot requires batch size 1, got {tuple(values.shape)}")
    return values[:, 0]


def _metrics(outputs, target: torch.Tensor, click_z: int | None = None) -> dict:
    native = _logits(outputs, "pred_masks_high_res").gt(0)
    raw = _logits(outputs, "stage2_raw_pred_masks_high_res").gt(0)
    result = {"native_dice": _dice(native, target), "raw_dice": _dice(raw, target)}
    if click_z is not None:
        result["clicked_slice_native_dice"] = _dice(native[click_z], target[click_z])
        result["clicked_slice_raw_dice"] = _dice(raw[click_z], target[click_z])
    return result


def _directional_clicked_metrics(state, target: torch.Tensor, click_z: int) -> dict | None:
    """Expose directional local response alongside fused-oracle behavior."""
    if state is None:
        return None
    return {
        "forward": _metrics(state.forward.outputs(), target, click_z),
        "reverse": _metrics(state.reverse.outputs(), target, click_z),
    }


def _oracle_error_membership(state, fused_hard: torch.Tensor, target: torch.Tensor, click) -> dict | None:
    """Verify that a fused-oracle click is an error in each directional view."""
    if state is None:
        return None
    z, y, x = int(click.z), int(click.y), int(click.x)
    expected_prediction = bool(int(click.label) < 0)  # FP needs foreground; FN needs background.
    forward = _logits(state.forward.outputs(), "pred_masks_high_res").gt(0)
    reverse = _logits(state.reverse.outputs(), "pred_masks_high_res").gt(0)
    return {
        "target": bool(target[z, y, x]),
        "fused_prediction": bool(fused_hard[z, y, x]),
        "forward_prediction": bool(forward[z, y, x]),
        "reverse_prediction": bool(reverse[z, y, x]),
        "expected_prediction_for_oracle_error": expected_prediction,
        "fused_is_oracle_error": bool(fused_hard[z, y, x]) == expected_prediction,
        "forward_is_oracle_error": bool(forward[z, y, x]) == expected_prediction,
        "reverse_is_oracle_error": bool(reverse[z, y, x]) == expected_prediction,
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _summarize_clicks(patients: list[dict]) -> dict:
    """Aggregate local and whole-volume correction behavior for the pilot."""
    rows = [
        row
        for patient in patients
        for placement in patient["placements"]
        for row in placement["clicks"]
        if row.get("click") is not None
    ]

    def summarize(group: list[dict]) -> dict:
        local = [float(row["delta_native_clicked_slice"]) for row in group]
        whole = [float(row["delta_native_whole"]) for row in group]
        return {
            "click_count": len(group),
            "mean_local_delta_native": _mean(local),
            "local_improvement_rate": _mean([float(value > 0.0) for value in local]),
            "mean_whole_volume_delta_native": _mean(whole),
            "whole_volume_improvement_rate": _mean([float(value > 0.0) for value in whole]),
        }

    fp = [row for row in rows if row["click"]["error_type"] == "FP"]
    fn = [row for row in rows if row["click"]["error_type"] == "FN"]
    first = [row for row in rows if not row["is_repeat_frame"]]
    repeat = [row for row in rows if row["is_repeat_frame"]]
    return {
        "all_clicks": summarize(rows),
        "by_error_type": {"FP": summarize(fp), "FN": summarize(fn)},
        "by_frame_visit": {
            "first_click_on_frame": summarize(first),
            "repeat_click_on_frame": summarize(repeat),
        },
    }


def _load_model(cli, device):
    model, _ = build_model(cli.model_cfg, cli.init_ckpt, device, _model_args(cli))
    state = torch.load(str(cli.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Invalid checkpoint: {cli.checkpoint}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={incompatible.missing_keys[:8]}, "
                           f"unexpected={incompatible.unexpected_keys[:8]}")
    model.eval()
    return model


def _state_factory():
    """Recognize persistent native/adapted implementations, otherwise legacy."""
    import stage2_tracking as tracking
    if hasattr(tracking, "OfficialMultiFrameBidirectionalState"):
        return "official_multiframe", tracking.OfficialMultiFrameBidirectionalState, tracking
    if hasattr(tracking, "MemoryDecoupledBidirectionalState"):
        return "memory_decoupled", tracking.MemoryDecoupledBidirectionalState, tracking
    if hasattr(tracking, "NativeBidirectionalState"):
        return "native_persistent", tracking.NativeBidirectionalState, tracking
    return "legacy_full_history", None, tracking


def _legacy_initial(tracking, model, batch, prompts, backbone):
    previous = torch.zeros(
        (int(batch.num_frames), *batch.masks.shape[-2:]),
        dtype=torch.bool, device=batch.masks.device,
    )
    outputs = tracking.bidirectional_mixed_outputs(model, batch, prompts, previous, [], backbone)
    return outputs, _logits(outputs, "pred_masks_high_res").gt(0), []


def _run_placement(kind, state_cls, tracking, model, batch, prompts, spacing) -> tuple[list[dict], list[dict]]:
    target = _gt(batch)
    if kind == "legacy_full_history":
        backbone = model.forward_image(batch.flat_img_batch)
        outputs, previous, clicks = _legacy_initial(tracking, model, batch, prompts, backbone)
        state = None
    else:
        backbone = model.forward_image(batch.flat_img_batch)
        state = state_cls(model, batch, prompts, backbone)
        outputs = state.outputs()
        previous = _logits(outputs, "pred_masks_high_res").gt(0)
        clicks = []
    curve = [{"round": 0, **_metrics(outputs, target)}]
    click_rows: list[dict] = []
    clicked_frames: set[int] = set()
    for round_index in range(1, 6):
        click = sample_correction_point(
            target.detach().cpu().numpy(), previous.detach().cpu().numpy(), spacing,
            "validation", exclude_slices=prompts,
        )
        if click is None:
            curve.append({"round": round_index, **_metrics(outputs, target)})
            click_rows.append({"round": round_index, "click": None, "plateau": True})
            continue
        before = _metrics(outputs, target, int(click.z))
        directional_before = _directional_clicked_metrics(state, target, int(click.z))
        oracle_membership = _oracle_error_membership(state, previous, target, click)
        is_repeat_frame = int(click.z) in clicked_frames
        clicked_frames.add(int(click.z))
        if kind == "legacy_full_history":
            clicks.append(click)
            outputs = tracking.bidirectional_mixed_outputs(
                model, batch, prompts, previous, clicks, backbone
            )
        else:
            state.add_click(click)
            outputs = state.outputs()
        previous = _logits(outputs, "pred_masks_high_res").gt(0)
        after = _metrics(outputs, target, int(click.z))
        directional_after = _directional_clicked_metrics(state, target, int(click.z))
        curve.append({"round": round_index, **_metrics(outputs, target)})
        click_rows.append({
            "round": round_index, "click": click.__dict__,
            "clicked_slice_before": before, "clicked_slice_after": after,
            "directional_clicked_slice_before": directional_before,
            "directional_clicked_slice_after": directional_after,
            "oracle_error_membership_before": oracle_membership,
            "is_repeat_frame": is_repeat_frame,
            "delta_native_clicked_slice": (
                after["clicked_slice_native_dice"] - before["clicked_slice_native_dice"]
            ),
            "delta_raw_clicked_slice": (
                after["clicked_slice_raw_dice"] - before["clicked_slice_raw_dice"]
            ),
            "delta_native_whole": after["native_dice"] - before["native_dice"],
            "delta_raw_whole": after["raw_dice"] - before["raw_dice"],
        })
    return curve, click_rows


def main() -> None:
    cli = _args()
    device = torch.device(cli.device)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(cli.validation_plan.read_text(encoding="utf-8"))["folds"][str(cli.fold)]
    patient_dirs = sorted(p for p in cli.data_root.iterdir() if p.is_dir() and p.name.startswith("p_"))
    fold = next(item for item in make_or_load_splits(patient_dirs, 5, 20260909, cli.split_path)
                if int(item["fold"]) == cli.fold)
    selected = sorted(fold["val"], key=lambda p: int(p.name.rsplit("_", 1)[1]))[:cli.patient_count]
    if len(selected) != cli.patient_count:
        raise ValueError(f"Only {len(selected)} validation patients available")
    kind, state_cls, tracking = _state_factory()
    model = _load_model(cli, device)
    report = {
        "checkpoint": str(cli.checkpoint), "fold": cli.fold, "patient_ids": [],
        "patient_selection": "first patient-count IDs in fixed fold validation split",
        "protocol": "K=3, both fixed placements, sequential validation oracle, D0..D5",
        "tracking_mode": kind,
        "raw_native_note": (
            "separate raw decoder and native gated outputs" if kind != "legacy_full_history"
            else "legacy full-history code has one decoder-logit output; native/raw are identical"
        ),
        "patients": [],
    }
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for patient_dir in selected:
            patient = int(patient_dir.name.rsplit("_", 1)[1])
            batch = next(iter(DataLoader(RectalCTVVolumeDataset([patient_dir]), batch_size=1, collate_fn=collate_one)))
            batch = batch.to(device, non_blocking=True)
            placements = plan[str(patient)]["placements"]["3"]
            if len(placements) != 2:
                raise ValueError(f"Patient {patient}: expected two K=3 placements")
            result = {"patient_id": patient, "placements": []}
            for placement_index, placement in enumerate(placements):
                prompts = [int(v) for v in placement["prompt_frame_ids"]]
                curve, click_rows = _run_placement(
                    kind, state_cls, tracking, model, batch, prompts, _spacing(patient_dir)
                )
                result["placements"].append({
                    "placement": placement_index, "prompt_frames": prompts,
                    "curve": curve, "clicks": click_rows,
                })
            for t in range(6):
                result[f"D{t}_native"] = sum(x["curve"][t]["native_dice"] for x in result["placements"]) / 2
                result[f"D{t}_raw"] = sum(x["curve"][t]["raw_dice"] for x in result["placements"]) / 2
            report["patient_ids"].append(patient)
            report["patients"].append(result)
            print(json.dumps({"patient_id": patient, **{k: result[k] for k in result if k.startswith("D")}}, sort_keys=True), flush=True)
    for field in ("native", "raw"):
        report[f"mean_D0_D5_{field}"] = {
            f"D{t}": sum(p[f"D{t}_{field}"] for p in report["patients"]) / len(report["patients"])
            for t in range(6)
        }
    report["click_behavior_summary"] = _summarize_clicks(report["patients"])
    (cli.output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (cli.output_dir / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as f:
        columns = ["patient_id"] + [f"D{t}_{view}" for view in ("native", "raw") for t in range(6)]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for patient in report["patients"]:
            writer.writerow({key: patient[key] for key in columns})
    print(json.dumps(report["mean_D0_D5_native"], sort_keys=True), flush=True)
    print(json.dumps(report["mean_D0_D5_raw"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
