#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2 single-prompt layer search on 3D NIfTI volumes (Z as video frames).

For each patient:
1) enumerate all GT-positive slices as candidate prompt layer (K=1),
2) run SAM2 video inference with exactly one GT-mask prompt slice,
3) compute 3D Dice against GT,
4) save per-layer records + best layer summary to Excel (two sheets),
5) save best-layer predicted mask as NIfTI.

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
OUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_1")
BEST_MASK_DIR = OUT_ROOT / "best_mask"
OUT_XLSX = OUT_ROOT / "prompt_layer_search.xlsx"

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

# Use the SAME yaml + pretrained checkpoint as the reference script.
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


def relative_pos_upper_to_lower(slice_id, lower_id, upper_id):
    """
    Relative position in [0,1] where:
    - close to upper bound -> small value (upper -> 0.0)
    - close to lower bound -> large value (lower -> 1.0)
    """
    if upper_id == lower_id:
        return 0.0
    return float((upper_id - slice_id) / (upper_id - lower_id))


# ======================================================
# =================== SAM2 Inference ====================
# ======================================================


@torch.no_grad()
def infer_with_single_prompt_slice(predictor, frame_dir, gt_zyx, prompt_slice):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    prompt_mask = (gt_zyx[prompt_slice] > 0).astype(np.uint8)
    if prompt_mask.sum() == 0:
        raise RuntimeError(f"Prompt slice {prompt_slice} is empty in GT.")

    predictor.add_new_mask(
        inference_state=state,
        frame_idx=int(prompt_slice),
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
            f"[INFO] {pdir.name} ({patient_id}) | candidates={len(candidate_slices)} "
            f"| lower={lower_id} upper={upper_id}"
        )

        best_dice = -1.0
        best_slice = None
        best_rel = None
        best_pred = None

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            for s in candidate_slices:
                pred = infer_with_single_prompt_slice(
                    predictor=predictor,
                    frame_dir=tmp_dir,
                    gt_zyx=gt_zyx,
                    prompt_slice=int(s),
                )
                dice = dice_3d(pred, gt_zyx)
                rel = relative_pos_upper_to_lower(int(s), lower_id, upper_id)

                row = {
                    "Patient_ID": patient_id,
                    "Prompt_Slice_ID": int(s),
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Prompt_Rel_Pos_UpperToLower": rel,
                    "Dice3D": dice,
                }
                all_rows.append(row)

                if (dice > best_dice) or (
                    abs(dice - best_dice) <= 1e-12 and best_rel is not None and rel < best_rel
                ):
                    best_dice = dice
                    best_slice = int(s)
                    best_rel = rel
                    best_pred = pred.copy()

            if best_slice is None or best_pred is None:
                print(f"[WARN] No valid prompt result for {patient_id}")
                continue

            write_mask_like(best_pred, img_sitk, out_mask_path)
            print(
                f"[OK] Best {patient_id}: slice={best_slice}, rel={best_rel:.4f}, "
                f"dice={best_dice:.4f} -> {out_mask_path}"
            )

            best_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Best_Prompt_Slice_ID": best_slice,
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Best_Prompt_Rel_Pos_UpperToLower": best_rel,
                    "Best_Dice3D": best_dice,
                }
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Sort outputs by numeric patient id.
    def _pid_key(row):
        mm = re.search(r"(\d+)$", str(row["Patient_ID"]))
        return int(mm.group(1)) if mm else 10**9

    all_rows.sort(key=lambda r: (_pid_key(r), r["Prompt_Slice_ID"]))
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
                int(r["Prompt_Slice_ID"]),
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                round(float(r["Prompt_Rel_Pos_UpperToLower"]), 6),
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
                int(r["Best_Prompt_Slice_ID"]),
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                round(float(r["Best_Prompt_Rel_Pos_UpperToLower"]), 6),
                round(float(r["Best_Dice3D"]), 6),
            ]
        )

    wb.save(str(OUT_XLSX))
    print(f"[DONE] Excel saved: {OUT_XLSX}")
    print(f"[DONE] Best masks saved in: {BEST_MASK_DIR}")


if __name__ == "__main__":
    main()
