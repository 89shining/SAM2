#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment_core import hd95_asd, uniform_prompt_indices
from io_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    RectalCTVVolumeDataset,
    build_model,
    ctv_case_name,
    list_patient_dirs,
    load_checkpoint,
    set_global_seed,
    write_model_evaluation_excel,
    write_prompt_record_excel,
)
from loops import unwrap_model
from bidirectional_tracking import bidirectional_outputs
from training.utils.data_utils import collate_fn


def collate_one(batch):
    return collate_fn(batch, dict_key="rectal_ctv_test")


def parse_list(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x.strip()) for x in part.split("-", 1)]
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def read_spacing_zyx(mask_path: Path):
    img = sitk.ReadImage(str(mask_path))
    spacing_xyz = img.GetSpacing()
    return (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))


def dice_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(pred, gt).sum() + eps) / (denom + eps))


@torch.no_grad()
def predict_probs_for_ckpt(args, model_x: int, fold: int, k: int, patient_dirs: list[Path], device: torch.device):
    ckpt_path = args.train_output_root / f"Model_{model_x}" / f"fold_{fold}" / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        print(f"[WARN] missing checkpoint: {ckpt_path}")
        return None

    model, _ = build_model(args.model_cfg, args.init_ckpt, device, args)
    load_checkpoint(ckpt_path, model, device=device)
    model.train(False)
    core_model = unwrap_model(model)

    ds = RectalCTVVolumeDataset(
        patient_dirs,
        input_size=args.input_size,
        window_center=args.window_center,
        window_width=args.window_width,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_one)

    probs_by_patient = {}
    for batch, pdir in zip(loader, ds.samples):
        batch = batch.to(device, non_blocking=True)
        prompt_frames = uniform_prompt_indices(int(batch.num_frames), k)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp), dtype=args.amp_dtype):
            outputs = bidirectional_outputs(
                core_model,
                batch,
                prompt_frames=prompt_frames,
                forward_backbone_per_frame=args.forward_backbone_per_frame,
            )

        gt_img = sitk.ReadImage(str(pdir / "CTV.nii.gz"))
        gt = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
        logits = torch.stack([out["pred_masks_high_res"][:, 0] for out in outputs], dim=0)[:, 0]
        logits_orig = F.interpolate(
            logits.unsqueeze(1),
            size=gt.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        prob = torch.sigmoid(logits_orig).detach().cpu().numpy().astype(np.float32)
        probs_by_patient[ctv_case_name(pdir)] = prob
    return probs_by_patient


def write_mask_like(pred_zyx: np.ndarray, ref_path: Path, out_path: Path):
    ref = sitk.ReadImage(str(ref_path))
    out = sitk.GetImageFromArray((pred_zyx > 0).astype(np.uint8))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path))


def run_ensemble_eval(args, model_x: int, k: int, folds: list[int], patient_dirs: list[Path], device: torch.device):
    prob_sum = {}
    prob_count = {}
    used_folds = []

    for fold in folds:
        print(f"[ENSEMBLE] Model-{model_x} fold {fold} test_k={k}")
        probs = predict_probs_for_ckpt(args, model_x, fold, k, patient_dirs, device)
        if probs is None:
            continue
        used_folds.append(fold)
        for patient_name, prob in probs.items():
            if patient_name not in prob_sum:
                prob_sum[patient_name] = prob.astype(np.float32)
                prob_count[patient_name] = 1
            else:
                prob_sum[patient_name] += prob.astype(np.float32)
                prob_count[patient_name] += 1

    rows = []
    for pdir in patient_dirs:
        patient_name = ctv_case_name(pdir)
        if patient_name not in prob_sum:
            print(f"[WARN] no ensemble prediction for {patient_name}, Model-{model_x}, k={k}")
            continue
        prob = prob_sum[patient_name] / float(prob_count[patient_name])
        pred = (prob >= 0.5).astype(np.uint8)

        gt_img = sitk.ReadImage(str(pdir / "CTV.nii.gz"))
        gt = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
        prompt_frames = uniform_prompt_indices(gt.shape[0], k)
        unprompted = [t for t in range(gt.shape[0]) if t not in set(prompt_frames)]
        spacing_zyx = read_spacing_zyx(pdir / "CTV.nii.gz")
        whole_hd95, whole_asd = hd95_asd(pred, gt, spacing_zyx)
        if unprompted:
            pred_un = pred[unprompted]
            gt_un = gt[unprompted]
            un_hd95, un_asd = hd95_asd(pred_un, gt_un, spacing_zyx)
            un_dice = dice_np(pred_un, gt_un)
        else:
            un_hd95, un_asd, un_dice = float("nan"), float("nan"), float("nan")

        if args.save_predictions:
            write_mask_like(
                pred,
                pdir / "CTV.nii.gz",
                args.output_root / f"Model_{model_x}" / f"Prompt_{k}" / f"{patient_name}.nii.gz",
            )

        rows.append(
            {
                "model_x": model_x,
                "test_k": k,
                "ensemble_folds": ";".join(str(x) for x in used_folds),
                "num_ensemble_folds": prob_count[patient_name],
                "patient": patient_name,
                "prompt_frames": ";".join(str(x) for x in prompt_frames),
                "unprompted_dice_3d": un_dice,
                "unprompted_hd95": un_hd95,
                "unprompted_asd": un_asd,
                "whole_dice_3d": dice_np(pred, gt),
                "whole_hd95": whole_hd95,
                "whole_asd": whole_asd,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser("Test SAM2-LoRA prompt-number experiment across all folds")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--models", type=str, default="2-6")
    parser.add_argument("--test-ks", type=str, default="2-6")
    parser.add_argument("--folds", type=str, default="0-4")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--forward-backbone-per-frame", action="store_true")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    args = parser.parse_args()

    args.amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if args.output_root is None:
        args.output_root = args.data_root.parent / "TestResults"
    args.output_root = args.output_root.resolve()
    if args.train_output_root is None:
        args.train_output_root = args.data_root.parent / "TrainResults"
    args.train_output_root = args.train_output_root.resolve()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_global_seed(args.seed)

    patient_dirs = list_patient_dirs(args.data_root / "test_nii")
    test_ks = parse_list(args.test_ks)
    write_prompt_record_excel(args.output_root / "test_prompt_frames.xlsx", patient_dirs, test_ks, "test")
    folds = parse_list(args.folds)
    for model_x in parse_list(args.models):
        model_rows = []
        for k in parse_list(args.test_ks):
            print(f"[TEST ENSEMBLE] Model-{model_x} test_k={k} folds={folds}")
            rows = run_ensemble_eval(args, model_x, k, folds, patient_dirs, device)
            model_rows.extend(rows)
        write_model_evaluation_excel(
            args.output_root / f"Model_{model_x}" / f"Model_{model_x}_evaluation.xlsx",
            model_rows,
            test_ks,
        )

    print(f"[DONE] results saved to {args.output_root}")


if __name__ == "__main__":
    main()
