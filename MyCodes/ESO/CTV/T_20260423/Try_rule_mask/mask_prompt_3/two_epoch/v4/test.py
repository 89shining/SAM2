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


def choose_middle_midpoint(lower_id: int, upper_id: int) -> int:
    return int((int(lower_id) + int(upper_id)) // 2)


def relative_pos_upper_to_lower(slice_id: int, lower_id: int, upper_id: int):
    if upper_id == lower_id:
        return 0.0
    return float((upper_id - slice_id) / (upper_id - lower_id))


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

    for frame_idx in processing_order:
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
                prev_sam_mask_logits = prev_sam_mask_logits_by_frame.get(int(frame_idx), None)
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
def infer_two_stage_midpoint_with_prev_mask(
    predictor,
    frame_dir: Path,
    gt_zyx: np.ndarray,
    lower_id: int,
    upper_id: int,
    middle_id: int,
    obj_id: int,
):
    z, h, w = gt_zyx.shape
    all_frames = list(range(z))

    # -------------------------
    # Stage-1: upper/lower only -> P0
    # -------------------------
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    stage1_prompt_ids = [int(upper_id)]
    if int(lower_id) != int(upper_id):
        stage1_prompt_ids.append(int(lower_id))
    stage1_prompt_ids = sorted(list(dict.fromkeys(stage1_prompt_ids)))
    for sid in stage1_prompt_ids:
        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=obj_id,
            mask=prompt_mask,
        )
    pred_stage1 = np.zeros((z, h, w), dtype=np.uint8)
    processing_order_stage1 = stage1_prompt_ids + [t for t in all_frames if t not in set(stage1_prompt_ids)]
    pred_stage1, pred_logits_stage1 = propagate_in_custom_order(
        predictor=predictor,
        state=state,
        processing_order=processing_order_stage1,
        obj_id=obj_id,
        pred_init=pred_stage1,
        prev_sam_mask_logits_by_frame=None,
    )

    # -------------------------
    # Stage-2: fresh state + upper/lower/middle prompts + P0 as previous mask
    # -------------------------
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    stage2_prompt_ids = [int(upper_id)]
    if int(lower_id) != int(upper_id):
        stage2_prompt_ids.append(int(lower_id))
    if int(middle_id) not in stage2_prompt_ids:
        stage2_prompt_ids.append(int(middle_id))
    stage2_prompt_ids = sorted(list(dict.fromkeys(stage2_prompt_ids)))

    for sid in stage2_prompt_ids:
        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(sid),
            obj_id=obj_id,
            mask=prompt_mask,
        )

    processing_order_stage2 = stage2_prompt_ids + [t for t in all_frames if t not in set(stage2_prompt_ids)]
    pred_stage2 = np.zeros((z, h, w), dtype=np.uint8)
    pred_stage2, _ = propagate_in_custom_order(
        predictor=predictor,
        state=state,
        processing_order=processing_order_stage2,
        obj_id=obj_id,
        pred_init=pred_stage2,
        prev_sam_mask_logits_by_frame=pred_logits_stage1,
    )

    return pred_stage1, pred_stage2


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
    parser = argparse.ArgumentParser("Test SAM2 with two-pass midpoint prompts and detached previous-mask refinement")
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
    parser.add_argument("--excel-name", type=str, default="two_pass_midpoint_prev_mask.xlsx")
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"test root not found: {args.test_root}")
    ckpt_path = resolve_ckpt(args.finetuned_ckpt, args.train_output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_stage1_dir = args.output_root / "mask_stage1_p0"
    mask_stage2_dir = args.output_root / "mask_stage2_p1_final"
    mask_stage1_dir.mkdir(parents=True, exist_ok=True)
    mask_stage2_dir.mkdir(parents=True, exist_ok=True)
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
    dice_stage1_all = []
    dice_stage2_all = []

    for pdir in patient_dirs:
        patient_id = patient_id_from_folder(pdir)
        out_mask_stage1 = mask_stage1_dir / f"{patient_id}.nii.gz"
        out_mask_stage2 = mask_stage2_dir / f"{patient_id}.nii.gz"
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
        middle_id = choose_middle_midpoint(lower_id=lower_id, upper_id=upper_id)

        rel = relative_pos_upper_to_lower(middle_id, lower_id, upper_id)
        print(
            f"[INFO] {patient_id} | fixed prompts: upper={upper_id}, lower={lower_id}, middle={middle_id}"
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_test_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir, args.window_center, args.window_width)
            pred_stage1, pred_stage2 = infer_two_stage_midpoint_with_prev_mask(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                middle_id=middle_id,
                obj_id=args.obj_id,
            )
            dice_stage1 = dice_3d(pred_stage1, gt_zyx)
            dice_stage2 = dice_3d(pred_stage2, gt_zyx)
            write_mask_like(pred_stage1, img_sitk, out_mask_stage1)
            write_mask_like(pred_stage2, img_sitk, out_mask_stage2)
            print(
                f"[OK] {patient_id}: stage1_dice={dice_stage1:.4f}, "
                f"stage2_dice={dice_stage2:.4f} | P0->{out_mask_stage1} | P1->{out_mask_stage2}"
            )
            dice_stage1_all.append(float(dice_stage1))
            dice_stage2_all.append(float(dice_stage2))

            all_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Prompt_Slice_ID": f"{upper_id},{lower_id},{middle_id}",
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Middle_ID": middle_id,
                    "Prompt_Rel_Pos_UpperToLower": rel,
                    "Dice3D_Stage1": dice_stage1,
                    "Dice3D_Stage2": dice_stage2,
                    "Mask_Stage1_Path": str(out_mask_stage1),
                    "Mask_Stage2_Path": str(out_mask_stage2),
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
            "Middle_ID",
            "Prompt_Rel_Pos_UpperToLower",
            "Dice3D_Stage1",
            "Dice3D_Stage2",
            "Mask_Stage1_Path",
            "Mask_Stage2_Path",
        ]
    )
    for r in all_rows:
        ws_all.append(
            [
                r["Patient_ID"],
                r["Prompt_Slice_ID"],
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                int(r["Middle_ID"]),
                round(float(r["Prompt_Rel_Pos_UpperToLower"]), 6),
                round(float(r["Dice3D_Stage1"]), 6),
                round(float(r["Dice3D_Stage2"]), 6),
                r["Mask_Stage1_Path"],
                r["Mask_Stage2_Path"],
            ]
        )

    wb.save(str(out_xlsx))
    if len(dice_stage1_all) > 0:
        print(
            f"[SUMMARY] mean_stage1_dice={np.mean(dice_stage1_all):.4f}, "
            f"mean_stage2_dice={np.mean(dice_stage2_all):.4f}"
        )
    print(f"[DONE] Excel saved: {out_xlsx}")
    print(f"[DONE] Stage1 masks (P0) saved in: {mask_stage1_dir}")
    print(f"[DONE] Stage2 masks (P1 final) saved in: {mask_stage2_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
