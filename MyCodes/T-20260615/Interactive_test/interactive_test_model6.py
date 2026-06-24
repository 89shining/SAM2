#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROMPT_NUMBER_DIR = CURRENT_DIR.parent / "Prompt_number"
if str(PROMPT_NUMBER_DIR) not in sys.path:
    sys.path.insert(0, str(PROMPT_NUMBER_DIR))

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from openpyxl import Workbook
from torch.utils.data import DataLoader

from bidirectional_tracking import precompute_backbone_out, track_in_order
from experiment_core import hd95_asd
from io_utils import (
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    RectalCTVVolumeDataset,
    build_model,
    ctv_case_name,
    list_patient_dirs,
    load_checkpoint,
    set_global_seed,
    write_model_evaluation_excel,
)
from test import dice_np, read_spacing_zyx, write_mask_like
from training.utils.data_utils import collate_fn


DEFAULT_ESO_DATA_ROOT = Path("/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260616_CTV/datanii")


def collate_one(batch):
    return collate_fn(batch, dict_key="interactive_ctv_test")


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


def initial_boundary_prompts(num_frames: int) -> list[int]:
    if num_frames <= 1:
        return [0]
    return [0, num_frames - 1]


def stack_probs_original(outputs: list[dict], out_hw: tuple[int, int]) -> np.ndarray:
    logits = torch.stack([out["pred_masks_high_res"][:, 0] for out in outputs], dim=0)[:, 0]
    logits_orig = F.interpolate(
        logits.unsqueeze(1),
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return torch.sigmoid(logits_orig).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_forward_backward_probs(core_model, batch, prompt_frames: list[int], out_hw: tuple[int, int], forward_backbone_per_frame: bool):
    base_backbone_out = precompute_backbone_out(core_model, batch, forward_backbone_per_frame)
    num_frames = int(batch.num_frames)
    outputs_forward = track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=list(range(num_frames)),
        track_in_reverse=False,
    )
    outputs_backward = track_in_order(
        core_model=core_model,
        base_backbone_out=base_backbone_out,
        batch=batch,
        prompt_frames=prompt_frames,
        frame_order=list(range(num_frames - 1, -1, -1)),
        track_in_reverse=True,
    )
    prob_forward = stack_probs_original(outputs_forward, out_hw)
    prob_backward = stack_probs_original(outputs_backward, out_hw)
    return prob_forward, prob_backward


def load_fold_model(args, fold: int, device: torch.device):
    ckpt_path = args.train_output_root / f"Model_{args.model_x}" / f"fold_{fold}" / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        print(f"[WARN] missing checkpoint: {ckpt_path}", flush=True)
        return None
    model, _ = build_model(args.model_cfg, args.init_ckpt, device, args)
    load_checkpoint(ckpt_path, model, device=device)
    model.train(False)
    return model


def predict_ensemble_round(args, folds: list[int], patient_dirs: list[Path], prompts_by_patient: dict[str, list[int]], device: torch.device):
    prob_sum = {}
    fwd_sum = {}
    bwd_sum = {}
    prob_count = {}
    used_folds = []

    for fold in folds:
        print(f"[ENSEMBLE] Model-{args.model_x} fold {fold}", flush=True)
        model = load_fold_model(args, fold, device)
        if model is None:
            continue
        used_folds.append(fold)
        core_model = model.module if hasattr(model, "module") else model
        ds = RectalCTVVolumeDataset(
            patient_dirs,
            input_size=args.input_size,
            window_center=args.window_center,
            window_width=args.window_width,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_one)
        for batch, pdir in zip(loader, ds.samples):
            patient_name = ctv_case_name(pdir)
            gt_img = sitk.ReadImage(str(pdir / "CTV.nii.gz"))
            gt = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)
            batch = batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp), dtype=args.amp_dtype):
                prob_forward, prob_backward = predict_forward_backward_probs(
                    core_model=core_model,
                    batch=batch,
                    prompt_frames=prompts_by_patient[patient_name],
                    out_hw=gt.shape[1:],
                    forward_backbone_per_frame=args.forward_backbone_per_frame,
                )
            prob = 0.5 * (prob_forward + prob_backward)
            if patient_name not in prob_sum:
                prob_sum[patient_name] = prob
                fwd_sum[patient_name] = prob_forward
                bwd_sum[patient_name] = prob_backward
                prob_count[patient_name] = 1
            else:
                prob_sum[patient_name] += prob
                fwd_sum[patient_name] += prob_forward
                bwd_sum[patient_name] += prob_backward
                prob_count[patient_name] += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    round_outputs = {}
    for pdir in patient_dirs:
        patient_name = ctv_case_name(pdir)
        if patient_name not in prob_sum:
            continue
        count = float(prob_count[patient_name])
        round_outputs[patient_name] = {
            "prob": prob_sum[patient_name] / count,
            "prob_forward": fwd_sum[patient_name] / count,
            "prob_backward": bwd_sum[patient_name] / count,
            "num_ensemble_folds": prob_count[patient_name],
            "ensemble_folds": ";".join(str(x) for x in used_folds),
        }
    return round_outputs


def per_slice_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(gt.shape[0]):
        vals.append(dice_np(pred[i], gt[i]))
    return np.asarray(vals, dtype=np.float32)


def select_next_prompt(strategy: str, round_output: dict, pred: np.ndarray, gt: np.ndarray, prompt_frames: list[int]) -> int | None:
    prompt_set = set(int(x) for x in prompt_frames)
    candidates = [i for i in range(gt.shape[0]) if i not in prompt_set]
    if not candidates:
        return None

    if strategy == "oracle":
        dice_vals = per_slice_dice(pred, gt)
        return min(candidates, key=lambda i: (float(dice_vals[i]), int(i)))

    if strategy == "uncertainty":
        diff = np.abs(round_output["prob_forward"] - round_output["prob_backward"])
        uncertainty = diff.reshape(diff.shape[0], -1).mean(axis=1)
        return max(candidates, key=lambda i: (float(uncertainty[i]), -int(i)))

    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_case(model_x: int, k: int, patient_name: str, prompt_frames: list[int], pred: np.ndarray, gt: np.ndarray, spacing_zyx, round_output: dict):
    unprompted = [t for t in range(gt.shape[0]) if t not in set(prompt_frames)]
    whole_hd95, whole_asd = hd95_asd(pred, gt, spacing_zyx)
    if unprompted:
        pred_un = pred[unprompted]
        gt_un = gt[unprompted]
        un_hd95, un_asd = hd95_asd(pred_un, gt_un, spacing_zyx)
        un_dice = dice_np(pred_un, gt_un)
    else:
        un_hd95, un_asd, un_dice = float("nan"), float("nan"), float("nan")
    return {
        "model_x": model_x,
        "test_k": k,
        "ensemble_folds": round_output["ensemble_folds"],
        "num_ensemble_folds": round_output["num_ensemble_folds"],
        "patient": patient_name,
        "prompt_frames": ";".join(str(x) for x in prompt_frames),
        "unprompted_dice_3d": un_dice,
        "unprompted_hd95": un_hd95,
        "unprompted_asd": un_asd,
        "whole_dice_3d": dice_np(pred, gt),
        "whole_hd95": whole_hd95,
        "whole_asd": whole_asd,
    }


def write_interactive_prompt_frames(path: Path, prompt_history: dict[str, dict[int, dict[str, list[int]]]], patient_dirs: list[Path]):
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = {}
    for pdir in patient_dirs:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / "CTV.nii.gz")))
        num_frames[ctv_case_name(pdir)] = int(gt.shape[0])

    wb = Workbook()
    wb.remove(wb.active)
    for strategy, by_k in prompt_history.items():
        for k in sorted(by_k):
            ws = wb.create_sheet(f"{strategy}_Prompt_{k}")
            ws.append(["strategy", "patient", "num_frames", "prompt_k", "prompt_frame_ids"])
            for patient_name in sorted(by_k[k]):
                prompts = by_k[k][patient_name]
                ws.append([strategy, patient_name, num_frames.get(patient_name, ""), int(k), ",".join(str(x) for x in prompts)])
    wb.save(path)


def run_strategy(args, strategy: str, folds: list[int], patient_dirs: list[Path], device: torch.device):
    print("=" * 100, flush=True)
    print(f"[STRATEGY] {strategy}", flush=True)
    print("=" * 100, flush=True)

    gt_by_patient = {}
    spacing_by_patient = {}
    prompts_by_patient = {}
    for pdir in patient_dirs:
        patient_name = ctv_case_name(pdir)
        gt = (sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / "CTV.nii.gz"))) > 0).astype(np.uint8)
        gt_by_patient[patient_name] = gt
        spacing_by_patient[patient_name] = read_spacing_zyx(pdir / "CTV.nii.gz")
        prompts_by_patient[patient_name] = initial_boundary_prompts(gt.shape[0])

    rows = []
    prompt_history = {strategy: {}}
    strategy_root = args.output_root / f"Model_{args.model_x}" / strategy

    for k in range(2, args.max_prompts + 1):
        print(f"[ROUND] strategy={strategy} prompt_k={k}", flush=True)
        prompt_history[strategy][k] = {name: list(prompts) for name, prompts in prompts_by_patient.items()}
        round_outputs = predict_ensemble_round(args, folds, patient_dirs, prompts_by_patient, device)

        next_prompts = {}
        for pdir in patient_dirs:
            patient_name = ctv_case_name(pdir)
            if patient_name not in round_outputs:
                print(f"[WARN] no prediction for {patient_name}, strategy={strategy}, k={k}", flush=True)
                continue

            gt = gt_by_patient[patient_name]
            prob = round_outputs[patient_name]["prob"]
            pred = (prob >= 0.5).astype(np.uint8)

            if args.save_predictions:
                write_mask_like(
                    pred,
                    pdir / "CTV.nii.gz",
                    strategy_root / f"Prompt_{k}" / f"{patient_name}.nii.gz",
                )

            rows.append(
                evaluate_case(
                    model_x=args.model_x,
                    k=k,
                    patient_name=patient_name,
                    prompt_frames=prompts_by_patient[patient_name],
                    pred=pred,
                    gt=gt,
                    spacing_zyx=spacing_by_patient[patient_name],
                    round_output=round_outputs[patient_name],
                )
            )

            if k < args.max_prompts:
                next_prompt = select_next_prompt(
                    strategy=strategy,
                    round_output=round_outputs[patient_name],
                    pred=pred,
                    gt=gt,
                    prompt_frames=prompts_by_patient[patient_name],
                )
                next_prompts[patient_name] = next_prompt

        if k < args.max_prompts:
            for patient_name, next_prompt in next_prompts.items():
                if next_prompt is None:
                    continue
                prompts = prompts_by_patient[patient_name]
                if int(next_prompt) not in set(prompts):
                    prompts.append(int(next_prompt))
                    prompts.sort()

    write_model_evaluation_excel(
        strategy_root / f"Model_{args.model_x}_{strategy}_evaluation.xlsx",
        rows,
        range(2, args.max_prompts + 1),
    )
    return prompt_history


def main():
    parser = argparse.ArgumentParser("Interactive Model-6 prompt testing with oracle/uncertainty prompt selection")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ESO_DATA_ROOT)
    parser.add_argument("--train-output-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--model-x", type=int, default=6)
    parser.add_argument("--max-prompts", type=int, default=6)
    parser.add_argument("--folds", type=str, default="0-4")
    parser.add_argument("--strategies", type=str, default="oracle,uncertainty")
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

    args.data_root = args.data_root.resolve()
    if args.train_output_root is None:
        args.train_output_root = args.data_root.parent / "TrainResults"
    args.train_output_root = args.train_output_root.resolve()
    if args.output_root is None:
        args.output_root = args.data_root.parent / "InteractiveTestResults"
    args.output_root = args.output_root.resolve()
    args.amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_global_seed(args.seed)
    patient_dirs = list_patient_dirs(args.data_root / "test_nii")
    folds = parse_list(args.folds)
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    all_prompt_history = {}
    for strategy in strategies:
        if strategy not in {"oracle", "uncertainty"}:
            raise ValueError(f"Unsupported strategy: {strategy}")
        prompt_history = run_strategy(args, strategy, folds, patient_dirs, device)
        all_prompt_history.update(prompt_history)

    write_interactive_prompt_frames(args.output_root / "interactive_prompt_frames.xlsx", all_prompt_history, patient_dirs)
    print(f"[DONE] interactive results saved to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
