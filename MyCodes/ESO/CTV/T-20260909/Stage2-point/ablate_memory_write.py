#!/usr/bin/env python
"""Read-only P0->P1 memory-write ablation for the adapted SAM2 checkpoint.

Variants
--------
baseline_global_native
    Current adapted implementation: a clicked frame becomes global conditioning
    memory and propagates normally.
local_only
    Replace only the clicked slice; do not write a new memory or propagate.
global_raw_present_memory
    Decode the click locally without memory attention, then write raw decoder
    mask logits to global conditioning memory while forcing the *memory encoder*
    object score to present.
directional_raw_present_memory
    The same local raw/present memory is a non-conditioning anchor.  It only
    updates frames after the click in the forward state and before it in the
    reverse state; initial GT-mask memories remain the only global cond frames.
"""
from __future__ import annotations

import argparse
import contextlib
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
from stage2_loops import _gt_volume_zyx, _whole_volume_dice
from stage2_tracking import (
    MemoryDecoupledBidirectionalState,
    NativeBidirectionalState,
    _attach_raw_logits,
    _capture_decoder,
    _point_input,
)
from training.utils.data_utils import collate_fn


STAGE1_RESULTS = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/"
    "Stage1-mask/TrainResults"
)


def collate_one(items):
    return collate_fn(items, dict_key="eso_ctv_memory_write_ablation")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--stage1-ckpt", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--patient-count", type=int, default=5)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "train")
    p.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    p.add_argument("--validation-plan", type=Path,
                   default=STAGE1_RESULTS / "validation_prompt_plan.json")
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    p.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _model_args(cli):
    return SimpleNamespace(
        model_cfg=cli.model_cfg, init_ckpt=cli.init_ckpt, input_size=512,
        lora_r=4, lora_alpha=16, lora_dropout=0.1,
        image_encoder_activation_checkpointing=True,
        fold=cli.fold, stage1_ckpt=cli.stage1_ckpt,
    )


def _load_model(cli, device):
    model, _ = build_model(cli.model_cfg, cli.init_ckpt, device, _model_args(cli))
    state = torch.load(str(cli.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Invalid checkpoint: {cli.checkpoint}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.eval()
    return model


def _spacing(patient_dir: Path) -> tuple[float, float, float]:
    sx, sy, sz = sitk.ReadImage(str(patient_dir / "image.nii.gz")).GetSpacing()
    return float(sz), float(sy), float(sx)


def _hard(outputs, field: str) -> torch.Tensor:
    actual = field if field in outputs[0] else "pred_masks_high_res"
    logits = torch.stack([out[actual][:, 0] for out in outputs], dim=0)
    return logits[:, 0].gt(0)


def _metrics(outputs, gt, z: int | None = None) -> dict:
    native, raw = _hard(outputs, "pred_masks_high_res"), _hard(outputs, "stage2_raw_pred_masks_high_res")
    result = {
        "native_whole": _whole_volume_dice(native, gt),
        "raw_whole": _whole_volume_dice(raw, gt),
    }
    if z is not None:
        result["native_clicked_slice"] = _whole_volume_dice(native[z], gt[z])
        result["raw_clicked_slice"] = _whole_volume_dice(raw[z], gt[z])
    return result


@contextlib.contextmanager
def _without_memory_attention(model):
    original = model._prepare_memory_conditioned_features

    def image_only(
        frame_idx, is_init_cond_frame, current_vision_feats,
        current_vision_pos_embeds, feat_sizes, output_dict, num_frames,
        track_in_reverse=False,
    ):
        del frame_idx, is_init_cond_frame, current_vision_pos_embeds, output_dict, num_frames, track_in_reverse
        feature = current_vision_feats[-1]
        batch_size, channels = feature.size(1), feature.size(2)
        height, width = feat_sizes[-1]
        return feature.permute(1, 2, 0).view(batch_size, channels, height, width)

    model._prepare_memory_conditioned_features = image_only
    try:
        yield
    finally:
        model._prepare_memory_conditioned_features = original


def _local_memory(state, click, *, source: str, force_present: bool):
    """Decode C1 once, then explicitly choose the mask/object-score for memory."""
    if source not in {"native", "raw"}:
        raise ValueError(f"Unknown memory source: {source}")
    canonical = state.canonical
    frame = int(click.z)
    prior = NativeBidirectionalState.fused_low_res_prior(state.forward, state.reverse, frame)
    canonical.point_history[frame].append(click)
    feats, pos, sizes = canonical._features(frame)
    captured = {}
    hook = state.canonical.model.sam_mask_decoder.register_forward_hook(_capture_decoder(captured))
    try:
        with _without_memory_attention(state.canonical.model):
            current = state.canonical.model.track_step(
                frame_idx=frame,
                is_init_cond_frame=False,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=sizes,
                point_inputs=_point_input(canonical.point_history[frame], prior.device),
                mask_inputs=None,
                gt_masks=None,
                frames_to_add_correction_pt=[],
                output_dict=canonical.output_dict,
                num_frames=int(canonical.batch.num_frames),
                track_in_reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=torch.clamp(prior, -32.0, 32.0),
            )
    finally:
        hook.remove()
    _attach_raw_logits(state.canonical.model, current, captured)
    score = current["multistep_object_score_logits"][-1]
    if force_present:
        # Avoid the no-object spatial embedding erasing a locally corrected
        # positive region before the next frame sees that correction.
        score = torch.full_like(score, 100.0)
    high_res = (
        current["stage2_raw_pred_masks_high_res"]
        if source == "raw"
        else current["pred_masks_high_res"]
    )
    mem_features, mem_pos = state.canonical.model._encode_new_memory(
        current_vision_feats=feats,
        feat_sizes=sizes,
        pred_masks_high_res=high_res,
        object_score_logits=score,
        is_mask_from_pts=True,
    )
    current["maskmem_features"] = mem_features
    current["maskmem_pos_enc"] = mem_pos
    return frame, current


def _local_raw_present(state, click):
    return _local_memory(state, click, source="raw", force_present=True)


def _global_raw_present(state, click):
    frame, current = _local_raw_present(state, click)
    canonical = state.canonical
    canonical.cond_frames.add(frame)
    canonical.output_dict["non_cond_frame_outputs"].pop(frame, None)
    canonical.output_dict["cond_frame_outputs"][frame] = current
    canonical.outputs_by_frame[frame] = current
    state.propagate_from_canonical(detach=True)
    return state.outputs()


def _directional_raw_present(state, click):
    frame, current = _local_raw_present(state, click)
    return _directional_write(state, frame, current)


def _directional_native(state, click):
    frame, current = _local_memory(state, click, source="native", force_present=False)
    return _directional_write(state, frame, current)


def _directional_raw(state, click):
    frame, current = _local_memory(state, click, source="raw", force_present=False)
    return _directional_write(state, frame, current)


def _directional_write(state, frame, current):
    # Keep initial masks as the only global conditioning frames.  The corrected
    # frame is a directional non-cond anchor and updates only its causal side.
    for view, reverse in ((state.forward, False), (state.reverse, True)):
        view.output_dict["non_cond_frame_outputs"][frame] = current
        view.outputs_by_frame[frame] = current
        order = range(frame - 1, -1, -1) if reverse else range(frame + 1, int(view.batch.num_frames))
        for index in order:
            if index not in view.cond_frames:
                view._run(index)
    return state.outputs()


def _local_only(state, click):
    frame, current = _local_raw_present(state, click)
    outputs = [dict(out) for out in state.outputs()]
    outputs[frame] = {key: value for key, value in current.items() if key != "obj_ptr"}
    return outputs


def _baseline(state, click):
    state.add_click(click)
    return state.outputs()


def _fresh(model, batch, prompts, backbone):
    return MemoryDecoupledBidirectionalState(model, batch, prompts, backbone)


def main() -> None:
    cli = _args()
    device = torch.device(cli.device)
    plan = json.loads(cli.validation_plan.read_text(encoding="utf-8"))["folds"][str(cli.fold)]
    dirs = sorted(p for p in cli.data_root.iterdir() if p.is_dir() and p.name.startswith("p_"))
    fold = next(x for x in make_or_load_splits(dirs, 5, 20260909, cli.split_path) if int(x["fold"]) == cli.fold)
    selected = sorted(fold["val"], key=lambda p: int(p.name.rsplit("_", 1)[1]))[:cli.patient_count]
    model = _load_model(cli, device)
    report = {"checkpoint": str(cli.checkpoint), "patient_ids": [], "variants": {}, "patients": []}
    variants = {
        "baseline_global_native": _baseline,
        "local_only": _local_only,
        "global_raw_present_memory": _global_raw_present,
        "directional_native_memory": _directional_native,
        "directional_raw_memory": _directional_raw,
        "directional_raw_present_memory": _directional_raw_present,
    }
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for directory in selected:
            patient = int(directory.name.rsplit("_", 1)[1])
            batch = next(iter(DataLoader(RectalCTVVolumeDataset([directory]), batch_size=1, collate_fn=collate_one))).to(device)
            gt = _gt_volume_zyx(batch).bool()
            backbone = model.forward_image(batch.flat_img_batch)
            row = {"patient_id": patient, "placements": []}
            for placement_index, placement in enumerate(plan[str(patient)]["placements"]["3"]):
                prompts = [int(v) for v in placement["prompt_frame_ids"]]
                base = _fresh(model, batch, prompts, backbone)
                p0 = base.outputs()
                previous = _hard(p0, "pred_masks_high_res")
                click = sample_correction_point(
                    gt.cpu().numpy(), previous.cpu().numpy(), _spacing(directory),
                    "validation", exclude_slices=prompts,
                )
                if click is None:
                    raise RuntimeError(f"Patient {patient} placement {placement_index}: no P0 correction point")
                placement_row = {
                    "placement": placement_index, "prompts": prompts, "click": click.__dict__,
                    "P0": _metrics(p0, gt, int(click.z)), "P1": {},
                }
                for name, operation in variants.items():
                    candidate = _fresh(model, batch, prompts, backbone)
                    outputs = operation(candidate, click)
                    placement_row["P1"][name] = _metrics(outputs, gt, int(click.z))
                row["placements"].append(placement_row)
            report["patient_ids"].append(patient)
            report["patients"].append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    for name in variants:
        report["variants"][name] = {}
        for metric in ("native_whole", "raw_whole", "native_clicked_slice", "raw_clicked_slice"):
            values = [p["placements"][j]["P1"][name][metric] for p in report["patients"] for j in range(2)]
            report["variants"][name][metric] = sum(values) / len(values)
    for metric in ("native_whole", "raw_whole", "native_clicked_slice", "raw_clicked_slice"):
        values = [p["placements"][j]["P0"][metric] for p in report["patients"] for j in range(2)]
        report["P0_mean_" + metric] = sum(values) / len(values)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({"P0": {k[8:]: v for k, v in report.items() if k.startswith("P0_mean_")},
                                  "variants": report["variants"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
