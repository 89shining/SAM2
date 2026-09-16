#!/usr/bin/env python
"""One-patient Stage-2 runtime probe; never optimizes or saves model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from stage1_bridge import (
    DEFAULT_DATA_ROOT, DEFAULT_INIT_CKPT, DEFAULT_MODEL_CFG, DEFAULT_SPLIT_PATH,
    RectalCTVVolumeDataset, build_model, make_or_load_splits,
)
from stage2_loops import (
    _gt_volume_zyx, _hard_mask_initialization, _initial_outputs, _terminal_loss,
    terminal_transition_from_history,
)
from stage2_tracking import OfficialMultiFrameBidirectionalState, hard_prediction, stacked_logits
from point_clicker import CorrectionPoint, sample_correction_point
from stage1_bridge import DiceBCELoss
from training.utils.data_utils import collate_fn


STAGE1_RESULTS = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/"
    "Stage1-mask/TrainResults"
)


def collate_one(items):
    return collate_fn(items, dict_key="eso_ctv_stage2_probe")


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, required=True)
    parser.add_argument("--patient-id", type=int)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "train")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--validation-plan", type=Path, default=STAGE1_RESULTS / "validation_prompt_plan.json")
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5],
        help="Correction budgets to probe; use '--budgets 0 1' for the preflight micro-smoke.",
    )
    parser.add_argument("--output", type=Path, default=Path("stage2_probe_report.json"))
    args = parser.parse_args()
    if any(budget < 0 or budget > 5 for budget in args.budgets):
        parser.error("--budgets values must be within 0..5")
    return args


def _model_args(cli):
    return SimpleNamespace(
        model_cfg=cli.model_cfg, init_ckpt=cli.init_ckpt, input_size=512,
        lora_r=4, lora_alpha=16, lora_dropout=0.1,
        image_encoder_activation_checkpointing=True,
        fold=cli.fold, stage1_ckpt=cli.stage1_ckpt,
    )


def _make_probe_model(cli, device):
    """Probe-local copy of model construction; avoids importing train.py by name."""
    model, _ = build_model(cli.model_cfg, cli.init_ckpt, device, _model_args(cli))
    state = torch.load(str(cli.stage1_ckpt), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Invalid Stage1 checkpoint: {cli.stage1_ckpt}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Stage1 checkpoint mismatch: missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return model


def _spacing(patient_dir: Path) -> tuple[float, float, float]:
    sx, sy, sz = sitk.ReadImage(str(patient_dir / "image.nii.gz")).GetSpacing()
    return float(sz), float(sy), float(sx)


def _execute_budget(model, batch, prompts, spacing_zyx, budget: int) -> dict:
    """Run fixed T in {0,1,3,5}; backward once and return interaction trace."""
    gt = _gt_volume_zyx(batch).bool()
    criterion = DiceBCELoss().to(gt.device)
    model.zero_grad(set_to_none=True)
    trace: list[dict] = []
    clicks: list[CorrectionPoint] = []
    if budget == 0:
        trace_start = len(trace)
        outputs = _initial_outputs(model, batch, prompts, trace=trace)
        for item in trace[trace_start:]:
            item["correction_round"] = 0
        loss, logits, loss_seg, loss_presence = _terminal_loss(criterion, outputs, gt, prompts)
    else:
        with torch.no_grad():
            cached = model.forward_image(batch.flat_img_batch)
            trajectory = OfficialMultiFrameBidirectionalState(model, batch, prompts, cached, trace)
            previous = hard_prediction(trajectory.outputs())
        for item in trace:
            item["correction_round"] = 0
        for round_index in range(1, budget + 1):
            click = sample_correction_point(
                gt.detach().cpu().numpy(), previous.detach().cpu().numpy(),
                spacing_zyx, "validation", exclude_slices=prompts,
            )
            if click is None:
                break
            clicks.append(click)
            if round_index < budget:
                with torch.no_grad():
                    trace_start = len(trace)
                    trajectory.add_click(click)
                    outputs = trajectory.outputs()
                    for item in trace[trace_start:]:
                        item["correction_round"] = round_index
                    previous = hard_prediction(outputs)
            else:
                trace_start = len(trace)
                terminal = trajectory.detached_terminal_snapshot()
                terminal.add_click(click)
                outputs = terminal.outputs()
                for item in trace[trace_start:]:
                    item["correction_round"] = round_index
                loss, logits, loss_seg, loss_presence = _terminal_loss(criterion, outputs, gt, prompts)
        else:
            pass
        if len(clicks) < budget:
            # An exact mask plateau is valid. Re-evaluate the last reachable state
            # if there is at least one click; T=0 already covers the no-click case.
            if not clicks:
                raise RuntimeError(
                    f"Budget T={budget} reached an exact P0 before any correction; "
                    "this probe cannot demonstrate the required correction backward pass."
                )
            outputs = terminal_transition_from_history(model, batch, prompts, clicks)
            loss, logits, loss_seg, loss_presence = _terminal_loss(criterion, outputs, gt, prompts)
    loss.backward()
    trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    trainable_with_grad = len(trainable_grads)
    finite_grad_tensors = sum(int(torch.isfinite(grad).all()) for grad in trainable_grads)
    nonzero_grad_tensors = sum(int(torch.any(grad != 0)) for grad in trainable_grads)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at T={budget}: {float(loss.detach())}")
    if nonzero_grad_tensors == 0:
        raise RuntimeError(f"No nonzero trainable gradient at T={budget}")
    contract_errors = []
    for item in trace:
        if item["is_initial_mask_frame"]:
            valid = (
                item["gt_masks"] is True
                and item["mask_inputs"] is True
                and item["point_inputs"] is False
                and item["prev_sam_mask_logits_shape"] is None
                and item["is_init_cond_frame"] is True
                and item["is_conditioning_output"] is True
            )
            reason = "invalid initial-mask frame state"
        elif item["is_point_frame"]:
            valid = (
                item["gt_masks"] is False
                and item["mask_inputs"] is False
                and item["point_inputs"] is True
                and item["prev_sam_mask_logits_shape"] is not None
                and item["is_init_cond_frame"] is False
                and item["is_conditioning_output"] is False
                and item["bypasses_memory_attention"] is False
            )
            reason = "invalid correction-frame state"
        else:
            valid = (
                item["gt_masks"] is False
                and item["mask_inputs"] is False
                and item["point_inputs"] is False
                and item["prev_sam_mask_logits_shape"] is None
                and item["is_init_cond_frame"] is False
                and item["is_conditioning_output"] is False
                and item["bypasses_memory_attention"] is False
            )
            reason = "invalid propagated-frame state"
        if not valid:
            contract_errors.append({"reason": reason, **item})
    prompt_set = set(map(int, prompts))
    for point in clicks:
        if int(point.z) in prompt_set:
            contract_errors.append({"reason": "correction point on initial-mask slice", **point.__dict__})
    if contract_errors:
        raise RuntimeError(f"Official multi-frame interaction contract failed: {contract_errors[:2]}")
    round_conditioning_frames = []
    for round_index in sorted({int(item["correction_round"]) for item in trace}):
        entries = [item for item in trace if int(item["correction_round"]) == round_index]
        directions = {}
        for direction in ("canonical", "forward", "reverse"):
            directions[direction] = [
                int(item["frame"]) for item in entries
                if item["direction"] == direction and (item["mask_inputs"] or item["point_inputs"])
            ]
        round_conditioning_frames.append({"round": round_index, **directions})
    return {
        "budget": budget,
        "clicks": [point.__dict__ for point in clicks],
        "loss": float(loss.detach()),
        "loss_seg": float(loss_seg.detach()),
        "loss_presence": float(loss_presence.detach()),
        "logit_shape": list(logits.shape),
        "trainable_parameters_with_grad": trainable_with_grad,
        "finite_gradient_tensors": finite_grad_tensors,
        "nonzero_gradient_tensors": nonzero_grad_tensors,
        "mixed_interaction_contract": "PASS",
        "round_conditioning_frames": round_conditioning_frames,
        "trace": trace,
    }


def main():
    cli = _args()
    device = torch.device(cli.device)
    all_dirs = sorted(path for path in cli.data_root.iterdir() if path.is_dir() and path.name.startswith("p_"))
    folds = make_or_load_splits(all_dirs, 5, 20260909, cli.split_path)
    fold = next(item for item in folds if int(item["fold"]) == cli.fold)
    plan = json.loads(cli.validation_plan.read_text(encoding="utf-8"))["folds"][str(cli.fold)]
    candidates = fold["val"]
    if cli.patient_id is not None:
        candidates = [p for p in candidates if int(p.name.rsplit("_", 1)[1]) == cli.patient_id]
    if len(candidates) != 1:
        raise ValueError("Select exactly one validation patient with --patient-id")
    patient_dir = candidates[0]
    patient = int(patient_dir.name.rsplit("_", 1)[1])
    batch = next(iter(DataLoader(RectalCTVVolumeDataset([patient_dir]), batch_size=1, collate_fn=collate_one)))
    batch = batch.to(device)
    prompts = [int(v) for v in plan[str(patient)]["placements"]["3"][0]["prompt_frame_ids"]]
    model = _make_probe_model(cli, device)
    model.train()
    report = {"patient_id": patient, "prompt_frames": prompts, "spacing_zyx": _spacing(patient_dir), "budgets": []}
    for budget in cli.budgets:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = _execute_budget(model, batch, prompts, report["spacing_zyx"], budget)
        if device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        report["budgets"].append(result)
        print(json.dumps({
            "patient_id": patient,
            "prompt_frames": prompts,
            "budget": budget,
            "clicks": result.get("clicks", []),
            "round_conditioning_frames": result.get("round_conditioning_frames", []),
            "loss": result.get("loss"),
            "trainable_parameters_with_grad": result.get("trainable_parameters_with_grad"),
            "finite_gradient_tensors": result.get("finite_gradient_tensors"),
            "nonzero_gradient_tensors": result.get("nonzero_gradient_tensors"),
            "peak_allocated_gib": result.get("peak_allocated_gib"),
            "peak_reserved_gib": result.get("peak_reserved_gib"),
            "mixed_interaction_contract": result.get("mixed_interaction_contract"),
        }), flush=True)
    cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {cli.output}", flush=True)


if __name__ == "__main__":
    main()
