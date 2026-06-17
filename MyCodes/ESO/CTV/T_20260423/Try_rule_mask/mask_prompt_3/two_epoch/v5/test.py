#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from openpyxl import Workbook
from PIL import Image

# keep local import robust
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[7]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2.build_sam import build_sam2_video_predictor


# ================= Defaults =================
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def window_to_uint8(img2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def save_frames_from_volume(vol_zyx: np.ndarray, out_dir: Path, wc: float, ww: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(vol_zyx.shape[0]):
        u8 = window_to_uint8(vol_zyx[i], wc, ww)
        rgb = np.stack([u8, u8, u8], axis=-1)
        Image.fromarray(rgb).save(out_dir / f"{i:05d}.jpg", quality=95)


def read_nii_zyx(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_mask_like(pred_zyx: np.ndarray, ref_img: sitk.Image, out_path: Path):
    pred_zyx = (pred_zyx > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(pred_zyx)
    out.SetSpacing(ref_img.GetSpacing())
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out, str(out_path))


def dice_3d(pred: np.ndarray, gt: np.ndarray, eps=1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def gt_positive_slices(gt_zyx: np.ndarray):
    non_empty = np.where(gt_zyx.reshape(gt_zyx.shape[0], -1).any(axis=1))[0]
    return [int(z) for z in non_empty.tolist()]


def build_seed_prompt_ids(gt_zyx: np.ndarray, frame_order, seed_prompt_mode: str):
    frame_order = [int(t) for t in frame_order]
    if len(frame_order) == 0:
        return []
    if seed_prompt_mode == "single":
        return [frame_order[0]]
    if seed_prompt_mode != "bounds":
        raise ValueError(f"Unsupported seed_prompt_mode: {seed_prompt_mode}")

    pos = gt_positive_slices(gt_zyx)
    if len(pos) == 0:
        return [frame_order[0]]
    lower_id = int(min(pos))
    upper_id = int(max(pos))
    cand = [lower_id, upper_id]
    return [t for t in frame_order if t in set(cand)] or [frame_order[0]]


@torch.no_grad()
def propagate_in_custom_order(
    predictor,
    state,
    processing_order,
    obj_id: int,
    pred_init: np.ndarray,
    prev_sam_mask_logits_by_frame=None,
):
    # Mirror predictor.propagate_in_video internals, but use explicit processing_order.
    predictor.propagate_in_video_preflight(state)
    obj_ids = state["obj_ids"]
    batch_size = predictor._get_obj_num(state)
    pred = pred_init.copy()
    pred_logits_by_frame = {}
    prev_sam_mask_logits_by_frame = prev_sam_mask_logits_by_frame or {}

    for order_idx, frame_idx in enumerate(processing_order):
        pred_masks_per_obj = [None] * batch_size
        for obj_idx in range(batch_size):
            obj_output_dict = state["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_output_dict["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = obj_output_dict[storage_key][frame_idx]
                device = state["device"]
                pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                if predictor.clear_non_cond_mem_around_input:
                    predictor._clear_obj_non_cond_mem_around_input(
                        state, frame_idx, obj_idx
                    )
            else:
                storage_key = "non_cond_frame_outputs"
                prev_sam_mask_logits = None
                if order_idx > 0:
                    prev_frame_idx = int(processing_order[order_idx - 1])
                    if prev_frame_idx in pred_logits_by_frame:
                        prev_sam_mask_logits = pred_logits_by_frame[prev_frame_idx]
                    elif prev_frame_idx in prev_sam_mask_logits_by_frame:
                        prev_sam_mask_logits = prev_sam_mask_logits_by_frame[prev_frame_idx]
                current_out, pred_masks = predictor._run_single_frame_inference(
                    inference_state=state,
                    output_dict=obj_output_dict,
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=prev_sam_mask_logits,
                )
                obj_output_dict[storage_key][frame_idx] = current_out

            state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
            pred_masks_per_obj[obj_idx] = pred_masks

        if len(pred_masks_per_obj) > 1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]
        _, video_res_masks = predictor._get_orig_video_res_output(state, all_pred_masks)

        for i, oid in enumerate(obj_ids):
            if int(oid) == obj_id:
                pred[int(frame_idx)] = (video_res_masks[i] > 0).cpu().numpy()
                pred_logits_by_frame[int(frame_idx)] = torch.clamp(
                    all_pred_masks[i:i + 1].detach(),
                    -32.0,
                    32.0,
                )
                break

    return pred, pred_logits_by_frame


@torch.no_grad()
def infer_bidirectional_fusion_with_prompt_gt(
    predictor,
    frame_dir: Path,
    gt_zyx: np.ndarray,
    obj_id: int,
    seed_prompt_mode: str,
):
    z, _, _ = gt_zyx.shape
    all_frames = list(range(z))
    frame_order_fwd = list(all_frames)
    frame_order_bwd = list(reversed(all_frames))

    def _run_direction(frame_order):
        state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(state)
        prompt_ids = build_seed_prompt_ids(gt_zyx, frame_order, seed_prompt_mode=seed_prompt_mode)
        for sid in prompt_ids:
            prompt_mask = (gt_zyx[int(sid)] > 0).astype(np.uint8)
            if prompt_mask.sum() == 0:
                raise RuntimeError(f"Prompt slice {sid} is empty in GT.")
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=int(sid),
                obj_id=obj_id,
                mask=prompt_mask,
            )
        pred_init = np.zeros(gt_zyx.shape, dtype=np.uint8)
        pred, pred_logits = propagate_in_custom_order(
            predictor=predictor,
            state=state,
            processing_order=frame_order,
            obj_id=obj_id,
            pred_init=pred_init,
            prev_sam_mask_logits_by_frame=None,
        )
        return pred, pred_logits, prompt_ids

    pred_forward, logits_forward, prompt_fwd = _run_direction(frame_order_fwd)
    pred_backward, logits_backward, prompt_bwd = _run_direction(frame_order_bwd)

    pred_fused = np.zeros_like(gt_zyx, dtype=np.uint8)
    for t in range(z):
        lf = logits_forward[int(t)]
        lb = logits_backward[int(t)]
        fused = 0.5 * (lf + lb)
        pred_fused[t] = (fused[0, 0] > 0).detach().cpu().numpy().astype(np.uint8)

    prompt_union = set(int(x) for x in (prompt_fwd + prompt_bwd))
    for t in prompt_union:
        pred_fused[int(t)] = (gt_zyx[int(t)] > 0).astype(np.uint8)

    return pred_forward, pred_backward, pred_fused, sorted(prompt_union)


def patient_id_from_folder(pdir: Path):
    m = re.search(r"\d+", pdir.name)
    if m is None:
        raise ValueError(f"Cannot parse patient id from folder name: {pdir.name}")
    return f"CTV_{int(m.group()):03d}"


def resolve_ckpt(finetuned_ckpt, train_output_root: Path) -> Path:
    """
    Priority:
    1) best fold from best_fold.txt under train_output_root
    2) explicit --finetuned-ckpt if provided and exists
    """
    best_fold_txt = train_output_root / "best_fold.txt"
    if best_fold_txt.exists():
        content = best_fold_txt.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"best_ckpt:\s*(.+)", content)
        if m:
            best_ckpt = Path(m.group(1).strip())
            if best_ckpt.exists():
                return best_ckpt

    if finetuned_ckpt is not None:
        finetuned_ckpt = Path(finetuned_ckpt)
        if finetuned_ckpt.exists():
            return finetuned_ckpt
        raise FileNotFoundError(f"--finetuned-ckpt not found: {finetuned_ckpt}")

    raise FileNotFoundError("No usable finetuned checkpoint found. Tried best_fold.txt and optional --finetuned-ckpt.")


def main():
    parser = argparse.ArgumentParser("Test SAM2 v5 bidirectional autoregressive fusion with prompt-layer GT")
    parser.add_argument("--test-root", type=Path, required=True, help="Separate test set root")
    parser.add_argument("--output-root", type=Path, required=True, help="Save root for masks/excel")
    parser.add_argument("--finetuned-ckpt", type=Path, default=None, help="Optional checkpoint for inference fallback")
    parser.add_argument("--train-output-root", type=Path, required=True, help="Training output root for auto resolving best fold ckpt")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--img-name", type=str, default="image.nii.gz")
    parser.add_argument("--gt-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--seed-prompt-mode",
        type=str,
        default="single",
        choices=["single", "bounds"],
        help="single: use first frame GT as seed per direction; bounds: use lower/upper GT seeds.",
    )
    parser.add_argument("--excel-name", type=str, default="v5_bidirectional_fusion.xlsx")
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"test root not found: {args.test_root}")
    ckpt_path = resolve_ckpt(args.finetuned_ckpt, args.train_output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_forward_dir = args.output_root / "mask_forward"
    mask_backward_dir = args.output_root / "mask_backward"
    mask_fused_dir = args.output_root / "mask_fused_final"
    mask_forward_dir.mkdir(parents=True, exist_ok=True)
    mask_backward_dir.mkdir(parents=True, exist_ok=True)
    mask_fused_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = args.output_root / args.excel_name

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(ckpt_path),
        device=device,
    )
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    patient_dirs = sorted([p for p in args.test_root.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(patient_dirs)} patients")

    all_rows = []
    dice_forward_all = []
    dice_backward_all = []
    dice_fused_all = []

    for pdir in patient_dirs:
        patient_id = patient_id_from_folder(pdir)
        out_mask_forward = mask_forward_dir / f"{patient_id}.nii.gz"
        out_mask_backward = mask_backward_dir / f"{patient_id}.nii.gz"
        out_mask_fused = mask_fused_dir / f"{patient_id}.nii.gz"
        img_path = pdir / args.img_name
        gt_path = pdir / args.gt_name

        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] Skip {pdir.name}: missing image or GT")
            continue

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)
        gt_zyx = (gt_zyx > 0).astype(np.uint8)

        if img_zyx.shape != gt_zyx.shape:
            print(f"[WARN] Skip {pdir.name}: shape mismatch img{img_zyx.shape} vs gt{gt_zyx.shape}")
            continue

        pos = gt_positive_slices(gt_zyx)
        if len(pos) == 0:
            print(f"[WARN] Skip {pdir.name}: GT has no positive slices")
            continue

        lower_id = int(min(pos))
        upper_id = int(max(pos))
        print(f"[INFO] {patient_id} | GT bounds: lower={lower_id}, upper={upper_id}")

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_test_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir, args.window_center, args.window_width)
            pred_forward, pred_backward, pred_fused, prompt_union = infer_bidirectional_fusion_with_prompt_gt(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                obj_id=args.obj_id,
                seed_prompt_mode=args.seed_prompt_mode,
            )
            dice_forward = dice_3d(pred_forward, gt_zyx)
            dice_backward = dice_3d(pred_backward, gt_zyx)
            dice_fused = dice_3d(pred_fused, gt_zyx)
            write_mask_like(pred_forward, img_sitk, out_mask_forward)
            write_mask_like(pred_backward, img_sitk, out_mask_backward)
            write_mask_like(pred_fused, img_sitk, out_mask_fused)
            print(
                f"[OK] {patient_id}: forward_dice={dice_forward:.4f}, "
                f"backward_dice={dice_backward:.4f}, fused_dice={dice_fused:.4f} | "
                f"FWD->{out_mask_forward} | BWD->{out_mask_backward} | FUSED->{out_mask_fused}"
            )
            dice_forward_all.append(float(dice_forward))
            dice_backward_all.append(float(dice_backward))
            dice_fused_all.append(float(dice_fused))

            all_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Prompt_Slice_ID": ",".join(str(x) for x in prompt_union),
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Seed_Prompt_Mode": args.seed_prompt_mode,
                    "Dice3D_Forward": dice_forward,
                    "Dice3D_Backward": dice_backward,
                    "Dice3D_Fused": dice_fused,
                    "Mask_Forward_Path": str(out_mask_forward),
                    "Mask_Backward_Path": str(out_mask_backward),
                    "Mask_Fused_Path": str(out_mask_fused),
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _pid_key(row):
        mm = re.search(r"(\d+)$", str(row["Patient_ID"]))
        return int(mm.group(1)) if mm else 10**9

    all_rows.sort(key=_pid_key)
    wb = Workbook()
    ws_all = wb.active
    ws_all.title = "Per_Patient"
    ws_all.append(
        [
            "Patient_ID",
            "Prompt_Slice_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Seed_Prompt_Mode",
            "Dice3D_Forward",
            "Dice3D_Backward",
            "Dice3D_Fused",
            "Mask_Forward_Path",
            "Mask_Backward_Path",
            "Mask_Fused_Path",
        ]
    )
    for r in all_rows:
        ws_all.append(
            [
                r["Patient_ID"],
                r["Prompt_Slice_ID"],
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                r["Seed_Prompt_Mode"],
                round(float(r["Dice3D_Forward"]), 6),
                round(float(r["Dice3D_Backward"]), 6),
                round(float(r["Dice3D_Fused"]), 6),
                r["Mask_Forward_Path"],
                r["Mask_Backward_Path"],
                r["Mask_Fused_Path"],
            ]
        )

    wb.save(str(out_xlsx))
    if len(dice_forward_all) > 0:
        print(
            f"[SUMMARY] mean_forward_dice={np.mean(dice_forward_all):.4f}, "
            f"mean_backward_dice={np.mean(dice_backward_all):.4f}, "
            f"mean_fused_dice={np.mean(dice_fused_all):.4f}"
        )
    print(f"[DONE] Excel saved: {out_xlsx}")
    print(f"[DONE] Forward masks saved in: {mask_forward_dir}")
    print(f"[DONE] Backward masks saved in: {mask_backward_dir}")
    print(f"[DONE] Fused final masks saved in: {mask_fused_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
