#!/usr/bin/env python
"""Decompose one adapted Stage-2 P0 -> C1 transition without optimization.

The report deliberately separates the image-only local correction from the
subsequent shared-memory forward/reverse propagation.  It also reports both
native (object-gated) and raw decoder Dice, so a propagation-only object gate
failure cannot be mistaken for a memory-propagation failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from point_clicker import sample_correction_point
from stage1_bridge import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    DEFAULT_SPLIT_PATH,
    RectalCTVVolumeDataset,
    build_model,
    make_or_load_splits,
)
from stage2_loops import _gt_volume_zyx, _whole_volume_dice
from stage2_tracking import (
    MemoryDecoupledBidirectionalState,
    NativeBidirectionalState,
    hard_prediction,
)
from training.utils.data_utils import collate_fn


STAGE1_RESULTS = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/"
    "Stage1-mask/TrainResults"
)


def collate_one(items):
    return collate_fn(items, dict_key="eso_ctv_first_correction_diagnostic")


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Adapted Stage-2 checkpoint to inspect (e.g. epoch-1 best.pth).")
    parser.add_argument("--patient-id", type=int, default=9)
    parser.add_argument("--placement", type=int, default=0, choices=(0, 1))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "train")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--validation-plan", type=Path,
                        default=STAGE1_RESULTS / "validation_prompt_plan.json")
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("first_correction_report.json"))
    return parser.parse_args()


def _model_args(cli):
    return SimpleNamespace(
        model_cfg=cli.model_cfg, init_ckpt=cli.init_ckpt, input_size=512,
        lora_r=4, lora_alpha=16, lora_dropout=0.1,
        image_encoder_activation_checkpointing=True,
        fold=cli.fold, stage1_ckpt=cli.stage1_ckpt,
    )


def _load_weights(model, checkpoint: Path) -> None:
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Invalid checkpoint: {checkpoint}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )


def _spacing(patient_dir: Path) -> tuple[float, float, float]:
    sx, sy, sz = sitk.ReadImage(str(patient_dir / "image.nii.gz")).GetSpacing()
    return float(sz), float(sy), float(sx)


def _hard_from_outputs(outputs, field: str) -> torch.Tensor:
    logits = torch.stack([out[field][:, 0] for out in outputs], dim=0)
    if logits.shape[1] != 1:
        raise ValueError(f"Expected batch size one, got {tuple(logits.shape)}")
    return logits[:, 0].gt(0.0)


def _metric(pred: torch.Tensor, gt: torch.Tensor, click_z: int | None = None) -> dict:
    row = {"whole_volume_dice": _whole_volume_dice(pred, gt)}
    if click_z is not None:
        row["clicked_slice_dice"] = _whole_volume_dice(pred[click_z], gt[click_z])
    return row


def _view_metrics(outputs, gt: torch.Tensor, click_z: int) -> dict:
    return {
        "native": _metric(_hard_from_outputs(outputs, "pred_masks_high_res"), gt, click_z),
        "raw": _metric(_hard_from_outputs(outputs, "stage2_raw_pred_masks_high_res"), gt, click_z),
    }


def main() -> None:
    cli = _args()
    device = torch.device(cli.device)
    all_dirs = sorted(p for p in cli.data_root.iterdir() if p.is_dir() and p.name.startswith("p_"))
    fold = next(p for p in make_or_load_splits(all_dirs, 5, 20260909, cli.split_path)
                if int(p["fold"]) == cli.fold)
    patient_dir = next(p for p in fold["val"] if int(p.name.rsplit("_", 1)[1]) == cli.patient_id)
    plan = json.loads(cli.validation_plan.read_text(encoding="utf-8"))["folds"][str(cli.fold)]
    prompts = [int(v) for v in plan[str(cli.patient_id)]["placements"]["3"][cli.placement]["prompt_frame_ids"]]
    batch = next(iter(DataLoader(RectalCTVVolumeDataset([patient_dir]), batch_size=1, collate_fn=collate_one)))
    batch = batch.to(device)
    model, _ = build_model(cli.model_cfg, cli.init_ckpt, device, _model_args(cli))
    _load_weights(model, cli.checkpoint)
    model.eval()
    gt = _gt_volume_zyx(batch).bool()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        backbone = model.forward_image(batch.flat_img_batch)
        state = MemoryDecoupledBidirectionalState(model, batch, prompts, backbone)
        p0_outputs = state.outputs()
        p0_hard = hard_prediction(p0_outputs)
        click = sample_correction_point(
            gt.detach().cpu().numpy(), p0_hard.detach().cpu().numpy(),
            _spacing(patient_dir), "validation", exclude_slices=prompts,
        )
        if click is None:
            raise RuntimeError("P0 has no residual error; a C1 decomposition is unavailable.")
        z = int(click.z)
        prior = NativeBidirectionalState.fused_low_res_prior(state.forward, state.reverse, z)
        state.canonical.add_click(
            click, previous_logits_override=prior,
            propagate=False, bypass_memory_attention=True,
        )
        local = state.canonical.outputs_by_frame[z]
        local_metrics = {
            "native": _metric(local["pred_masks_high_res"][:, 0].gt(0.0), gt[z], None),
            "raw": _metric(local["stage2_raw_pred_masks_high_res"][:, 0].gt(0.0), gt[z], None),
        }
        state.propagate_from_canonical(detach=True)
        forward_metrics = _view_metrics(state.forward.outputs(), gt, z)
        reverse_metrics = _view_metrics(state.reverse.outputs(), gt, z)
        fusion_metrics = _view_metrics(state.outputs(), gt, z)
        report = {
            "checkpoint": str(cli.checkpoint),
            "patient_id": cli.patient_id,
            "placement": cli.placement,
            "prompt_frames": prompts,
            "click": click.__dict__,
            "P0_fusion": _view_metrics(p0_outputs, gt, z),
            "P1_local_before_propagation": local_metrics,
            "P1_forward_after_propagation": forward_metrics,
            "P1_reverse_after_propagation": reverse_metrics,
            "P1_fusion_after_propagation": fusion_metrics,
        }
    cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote {cli.output}", flush=True)


if __name__ == "__main__":
    main()
