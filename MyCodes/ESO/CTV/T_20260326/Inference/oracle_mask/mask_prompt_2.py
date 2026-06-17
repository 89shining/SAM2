#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2 fixed-two-prompt inference on 3D NIfTI volumes (Z as video frames).

For each patient:
1) find GT-positive slice bounds: lower and upper,
2) run SAM2 video inference with exactly two GT-mask prompts (upper + lower),
3) compute 3D Dice against GT,
4) save records to Excel with the same table structure as prompt-layer search,
5) save predicted mask as NIfTI.

Note:
- Best prompt layer related fields are intentionally left blank.

Prediction naming rule:
  p_10 -> CTV_010.nii.gz
"""
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from openpyxl import Workbook
import torch
from PIL import Image

# Keep compatible with existing environment settings.
sys.path.append("/home/wusi/SAM2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.build_sam import build_sam2_video_predictor


# ======================================================
# ================== Path & Config =====================
# ======================================================

DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260108/Data83")
OUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_2")
BEST_MASK_DIR = OUT_ROOT / "best_mask"
OUT_XLSX = OUT_ROOT / "prompt_layer_search2.xlsx"

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

# Use the SAME yaml + pretrained checkpoint.
SAM2_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"  # "cuda" or "cpu"
OBJ_ID = 1

# CT window
WINDOW_CENTER = 40
WINDOW_WIDTH = 400


# ======================================================
# ======================= Utils =========================
# ======================================================


def window_to_uint8(img2d, wc, ww):
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def save_frames_from_volume(vol_zyx, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(vol_zyx.shape[0]):
        u8 = window_to_uint8(vol_zyx[i], WINDOW_CENTER, WINDOW_WIDTH)
        rgb = np.stack([u8, u8, u8], axis=-1)
        Image.fromarray(rgb).save(out_dir / f"{i:05d}.jpg", quality=95)


def read_nii_zyx(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_mask_like(pred_zyx, ref_img, out_path):
    pred_zyx = (pred_zyx > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(pred_zyx)
    out.SetSpacing(ref_img.GetSpacing())
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out, str(out_path))


def dice_3d(pred, gt, eps=1e-8):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def gt_positive_slices(gt_zyx):
    non_empty = np.where(gt_zyx.reshape(gt_zyx.shape[0], -1).any(axis=1))[0]
    return [int(z) for z in non_empty.tolist()]


# ======================================================
# =================== SAM2 Inference ====================
# ======================================================


@torch.no_grad()
def infer_with_two_prompt_slices(predictor, frame_dir, gt_zyx, lower_id, upper_id):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    # Add prompts on both bounds. If they are the same slice, add once.
    prompt_ids = [int(upper_id)]
    if int(lower_id) != int(upper_id):
        prompt_ids.append(int(lower_id))

    for sid in prompt_ids:
        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")

        predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=OBJ_ID,
            mask=prompt_mask,
        )

    z, h, w = gt_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)

    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == OBJ_ID:
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break

    return pred


# ======================================================
# ======================== Main =========================
# ======================================================


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    BEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)

    patient_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"[INFO] Found {len(patient_dirs)} patients")

    all_rows = []
    best_rows = []

    for pdir in patient_dirs:
        m = re.fullmatch(r"p_(\d+)", pdir.name)
        if m is None:
            print(f"[WARN] Skip invalid folder name: {pdir.name}")
            continue

        pid_num = int(m.group(1))
        patient_id = f"CTV_{pid_num:03d}"
        out_mask_path = BEST_MASK_DIR / f"{patient_id}.nii.gz"

        img_path = pdir / IMG_NAME
        gt_path = pdir / GT_NAME
        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] Skip {pdir.name}: missing image or GT")
            continue

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)
        gt_zyx = (gt_zyx > 0).astype(np.uint8)

        if img_zyx.shape != gt_zyx.shape:
            print(f"[WARN] Skip {pdir.name}: shape mismatch img{img_zyx.shape} vs gt{gt_zyx.shape}")
            continue

        candidate_slices = gt_positive_slices(gt_zyx)
        if len(candidate_slices) == 0:
            print(f"[WARN] Skip {pdir.name}: GT has no positive slices")
            continue

        lower_id = int(min(candidate_slices))
        upper_id = int(max(candidate_slices))

        print(
            f"[INFO] {pdir.name} ({patient_id}) | fixed prompts: upper={upper_id}, lower={lower_id}"
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            pred = infer_with_two_prompt_slices(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
            )
            dice = dice_3d(pred, gt_zyx)

            write_mask_like(pred, img_sitk, out_mask_path)
            print(f"[OK] {patient_id}: dice={dice:.4f} -> {out_mask_path}")

            # Keep sheet schema unchanged.
            all_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Prompt_Slice_ID": f"{upper_id},{lower_id}",
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Prompt_Rel_Pos_UpperToLower": None,
                    "Dice3D": dice,
                }
            )

            best_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Best_Prompt_Slice_ID": None,
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Best_Prompt_Rel_Pos_UpperToLower": None,
                    "Best_Dice3D": dice,
                }
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Sort outputs by numeric patient id.
    def _pid_key(row):
        mm = re.search(r"(\d+)$", str(row["Patient_ID"]))
        return int(mm.group(1)) if mm else 10**9

    all_rows.sort(key=_pid_key)
    best_rows.sort(key=_pid_key)

    wb = Workbook()

    ws_all = wb.active
    ws_all.title = "All_Prompt_Search"
    ws_all.append(
        [
            "Patient_ID",
            "Prompt_Slice_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Prompt_Rel_Pos_UpperToLower",
            "Dice3D",
        ]
    )
    for r in all_rows:
        ws_all.append(
            [
                r["Patient_ID"],
                r["Prompt_Slice_ID"],
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                None,
                round(float(r["Dice3D"]), 6),
            ]
        )

    ws_best = wb.create_sheet("Best_Per_Patient")
    ws_best.append(
        [
            "Patient_ID",
            "Best_Prompt_Slice_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Best_Prompt_Rel_Pos_UpperToLower",
            "Best_Dice3D",
        ]
    )
    for r in best_rows:
        ws_best.append(
            [
                r["Patient_ID"],
                None,
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                None,
                round(float(r["Best_Dice3D"]), 6),
            ]
        )

    wb.save(str(OUT_XLSX))
    print(f"[DONE] Excel saved: {OUT_XLSX}")
    print(f"[DONE] Masks saved in: {BEST_MASK_DIR}")


if __name__ == "__main__":
    main()
