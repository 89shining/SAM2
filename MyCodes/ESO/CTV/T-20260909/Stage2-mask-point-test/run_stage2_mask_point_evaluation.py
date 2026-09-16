#!/usr/bin/env python
"""Deterministic full Stage-2 evaluation: frozen Stage-1 masks plus P0..P5."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

STAGE2_DIR = Path(__file__).resolve().parents[1] / "Stage2-point"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from stage1_bridge import (
    DEFAULT_DATA_ROOT, DEFAULT_INIT_CKPT, DEFAULT_MODEL_CFG,
    RectalCTVVolumeDataset, build_model, list_patient_dirs,
)
from stage2_tracking import OfficialMultiFrameBidirectionalState, hard_prediction
from point_clicker import sample_correction_point
from experiment_core import hd95_asd, unprompted_slice_3d_dsc
from training.utils.data_utils import collate_fn

DEFAULT_PLAN = Path(
    "/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage1-lasso/"
    "TestResults/external_test_prompt_plans/stage1_prompt_plan.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "test")
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    p.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--image-encoder-activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-predictions", action="store_true")
    return p.parse_args()


def patient_id(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[1])


def collate_one(items):
    return collate_fn(items, dict_key="eso_ctv_stage2_test")


def make_batch(case: Path, device: torch.device):
    return collate_one([RectalCTVVolumeDataset([case])[0]]).to(device, non_blocking=True)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float((2 * (pred & gt).sum() + 1e-6) / (denom + 1e-6))


def endpoint_mae(pred: np.ndarray, gt: np.ndarray, reference: sitk.Image):
    if not pred.any():
        return None, None
    origin, step = reference.GetOrigin()[2], reference.GetDirection()[8] * reference.GetSpacing()[2]
    def ends(mask):
        values = origin + step * np.where(mask)[0]
        return float(values.max()), float(values.min())
    pred_sup, pred_inf = ends(pred); gt_sup, gt_inf = ends(gt)
    return abs(pred_sup - gt_sup), abs(pred_inf - gt_inf)


def describe(values):
    values = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(values):
        return {"n_valid": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {"n_valid": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75))}


def save_prediction(pred: np.ndarray, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(pred.astype(np.uint8)); image.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(image, str(path), True)


def load_plan(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 2 or doc.get("plan_type") != "deterministic":
        raise ValueError("Stage2 main test requires a schema-v2 deterministic frozen mask-prompt plan")
    return doc["patients"]


def main() -> None:
    a = parse_args(); device = torch.device(a.device)
    if a.fold != 0:
        raise ValueError("Only the available fold0 Stage2 checkpoint may be evaluated")
    plan, cases = load_plan(a.plan), {patient_id(x): x for x in list_patient_dirs(a.data_root)}
    if set(cases) != set(map(int, plan)):
        raise RuntimeError("External-test data and frozen prompt-plan patient IDs differ")
    if not a.checkpoint.is_file():
        raise FileNotFoundError(a.checkpoint)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("Checkpoint does not contain Stage2 model weights")
    ckpt_args = state.get("args", {})
    if isinstance(ckpt_args, dict) and "fold" in ckpt_args and int(ckpt_args["fold"]) != a.fold:
        raise RuntimeError("Stage2 checkpoint fold does not match --fold")
    model, model_stats = build_model(a.model_cfg, a.init_ckpt, device, a)
    model.load_state_dict(state["model"], strict=True); model.eval()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "run_manifest.json").write_text(json.dumps({
        "evaluation_scope": "external_test_stage2_mask_point", "patients": len(cases),
        "fold": a.fold, "checkpoint": str(a.checkpoint), "stage1_checkpoint": state.get("stage1_checkpoint"),
        "plan": str(a.plan), "mask_prompt": "frozen deterministic SI-stratified K1..K5",
        "point_prompt": "dynamic largest 26-connected FN/FP component; deterministic spacing-aware EDT maximum",
        "point_budgets": [0, 1, 2, 3, 4, 5], "initial_mask_slices_excluded_from_points": True,
        "model_stats": model_stats,
    }, indent=2), encoding="utf-8")
    rows, click_rows = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda"):
        for pid, case in sorted(cases.items()):
            batch = make_batch(case, device); gt_t = batch.masks[:, 0].bool(); gt = gt_t.cpu().numpy()
            reference = sitk.ReadImage(str(case / "CTV.nii.gz"))
            spacing = tuple(reversed(reference.GetSpacing()))
            backbone = model.forward_image(batch.flat_img_batch)
            for key, record in sorted(plan[str(pid)]["plans"].items(), key=lambda item: int(item[0][1:])):
                if not record["feasible"]:
                    raise RuntimeError(f"Infeasible frozen prompt plan p_{pid} {key}")
                prompts = [int(z) for z in record["slices"]]
                positive = set(np.flatnonzero(gt.reshape(gt.shape[0], -1).any(1)).astype(int).tolist())
                if not set(prompts).issubset(positive) or any(b - x < 2 for x, b in zip(prompts, prompts[1:])):
                    raise RuntimeError(f"Invalid frozen mask prompts p_{pid} {key}")
                interaction = OfficialMultiFrameBidirectionalState(model, batch, prompts, backbone)
                effective_t = 0
                for requested_t in range(6):
                    outputs = interaction.outputs(); pred = hard_prediction(outputs).detach().cpu().numpy().astype(bool)
                    empty = not pred.any()
                    hd95 = sup = inf = None
                    if not empty:
                        hd95, _ = hd95_asd(pred, gt, spacing); sup, inf = endpoint_mae(pred, gt, reference)
                    rows.append({"patient_id": pid, "fold": a.fold, "plan_key": key, "K": len(prompts),
                                 "requested_T": requested_t, "effective_T": effective_t,
                                 "prompt_slices": json.dumps(prompts), "whole_volume_dice": dice(pred, gt),
                                 "unprompted_dice": unprompted_slice_3d_dsc(outputs, batch.masks, prompts),
                                 "hd95_3d_mm": hd95, "prediction_empty": int(empty),
                                 "sup_endpoint_mae_mm": sup, "inf_endpoint_mae_mm": inf})
                    if a.save_predictions:
                        save_prediction(pred, reference, a.output_dir / "predictions" / key / f"T{requested_t}" / f"p_{pid:03d}.nii.gz")
                    print(f"[{key} T{requested_t}] p_{pid:03d} Dice={rows[-1]['whole_volume_dice']:.4f}", flush=True)
                    if requested_t == 5:
                        continue
                    click = sample_correction_point(gt, pred, spacing, "validation", exclude_slices=prompts)
                    if click is None:
                        click_rows.append({"patient_id": pid, "plan_key": key, "K": len(prompts), "round": requested_t + 1,
                                           "applied": 0, "reason": "exact_prediction"})
                        continue
                    interaction.add_click(click); effective_t += 1
                    click_rows.append({"patient_id": pid, "plan_key": key, "K": len(prompts), "round": requested_t + 1,
                                       "applied": 1, **click.__dict__})
    fields = list(rows[0]);
    with (a.output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    click_fields = sorted({field for row in click_rows for field in row})
    with (a.output_dir / "point_trace.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=click_fields); w.writeheader(); w.writerows(click_rows)
    metrics = ["whole_volume_dice", "hd95_3d_mm", "sup_endpoint_mae_mm", "inf_endpoint_mae_mm", "unprompted_dice"]
    summary = {"evaluation_scope": "external_test_stage2_mask_point", "checkpoint": str(a.checkpoint),
               "stage1_checkpoint": state.get("stage1_checkpoint"), "plan": str(a.plan), "n": len(rows),
               "primary_metrics": metrics[:4], "auxiliary_metrics": ["unprompted_dice"], "by_K_T": {}}
    for k in range(1, 6):
        for t in range(6):
            selected = [row for row in rows if row["K"] == k and row["requested_T"] == t]
            summary["by_K_T"][f"K{k}_T{t}"] = {"n": len(selected), "n_empty_prediction": sum(row["prediction_empty"] for row in selected),
                **{metric: describe([row[metric] for row in selected]) for metric in metrics}}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
