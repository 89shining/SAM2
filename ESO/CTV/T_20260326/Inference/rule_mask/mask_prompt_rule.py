#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2 fixed upper/lower + rule-based third-prompt inference on 3D NIfTI volumes.

Rules for the 3rd prompt (all selected from GT-positive middle layers: lower < z < upper):
1) Middle rule: choose the geometric middle layer; if two candidates, pick the one closer to upper.
2) Max-area rule: choose the layer with largest GT area; if tie, pick the one closer to upper.
3) Max-change rule: choose the layer with largest average area change to previous and next layer:
       score(z) = (|A(z)-A(z-1)| + |A(z)-A(z+1)|) / 2
   if tie, pick the one closer to upper.

Other pipeline behavior follows mask_prompt_3.py:
- read Data83/p_xxx/image.nii.gz and CTV.nii.gz
- fixed GT prompts on upper + lower
- run SAM2 inference
- compute 3D Dice
- save NIfTI masks
- save one Excel sheet summary
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
OUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/Try_rule_mask")

MID_MASK_DIR = OUT_ROOT / "rule_middle_mask"
MAXAREA_MASK_DIR = OUT_ROOT / "rule_maxarea_mask"
MAXCHANGE_MASK_DIR = OUT_ROOT / "rule_maxarea_change_mask"

OUT_XLSX = OUT_ROOT / "prompt_rule_summary.xlsx"

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

# Keep the same yaml + checkpoint.
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


def choose_middle_rule(middle_candidates):
    """
    Pick geometric middle. If there are two middle layers, pick the upper-side one.
    With upper_id > lower_id, 'closer to upper' means larger z id.
    """
    if len(middle_candidates) == 0:
        return None
    cands = sorted(int(x) for x in middle_candidates)
    return int(cands[len(cands) // 2])


def choose_maxarea_rule(gt_zyx, middle_candidates):
    """Pick layer with maximum GT area; tie -> closer to upper (larger z)."""
    if len(middle_candidates) == 0:
        return None

    cands = sorted(int(x) for x in middle_candidates)
    areas = {z: int((gt_zyx[z] > 0).sum()) for z in cands}
    max_area = max(areas.values())
    best = [z for z in cands if areas[z] == max_area]
    return int(max(best))


def choose_maxchange_rule(gt_zyx, middle_candidates):
    """
    Pick layer with largest average area change to previous and next slice:
      score(z) = (|A(z)-A(z-1)| + |A(z)-A(z+1)|) / 2
    tie -> closer to upper (larger z).
    """
    if len(middle_candidates) == 0:
        return None

    cands = sorted(int(x) for x in middle_candidates)
    zdim = gt_zyx.shape[0]
    area = [int((gt_zyx[z] > 0).sum()) for z in range(zdim)]

    scores = {}
    for z in cands:
        prev_z = max(0, z - 1)
        next_z = min(zdim - 1, z + 1)
        s = (abs(area[z] - area[prev_z]) + abs(area[z] - area[next_z])) / 2.0
        scores[z] = float(s)

    max_score = max(scores.values())
    best = [z for z in cands if abs(scores[z] - max_score) <= 1e-12]
    return int(max(best))


# ======================================================
# =================== SAM2 Inference ====================
# ======================================================


@torch.no_grad()
def infer_with_upper_lower_middle(predictor, frame_dir, gt_zyx, lower_id, upper_id, middle_id=None):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    prompt_ids = [int(upper_id)]
    if int(lower_id) != int(upper_id):
        prompt_ids.append(int(lower_id))
    if middle_id is not None:
        prompt_ids.append(int(middle_id))

    # Keep order stable and avoid accidental duplicates.
    prompt_ids = list(dict.fromkeys(prompt_ids))

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
    MID_MASK_DIR.mkdir(parents=True, exist_ok=True)
    MAXAREA_MASK_DIR.mkdir(parents=True, exist_ok=True)
    MAXCHANGE_MASK_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)

    patient_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"[INFO] Found {len(patient_dirs)} patients")

    rows = []

    for pdir in patient_dirs:
        m = re.fullmatch(r"p_(\d+)", pdir.name)
        if m is None:
            print(f"[WARN] Skip invalid folder name: {pdir.name}")
            continue

        pid_num = int(m.group(1))
        patient_id = f"CTV_{pid_num:03d}"

        out_mid_path = MID_MASK_DIR / f"{patient_id}.nii.gz"
        out_maxarea_path = MAXAREA_MASK_DIR / f"{patient_id}.nii.gz"
        out_maxchange_path = MAXCHANGE_MASK_DIR / f"{patient_id}.nii.gz"

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

        positive_slices = gt_positive_slices(gt_zyx)
        if len(positive_slices) == 0:
            print(f"[WARN] Skip {pdir.name}: GT has no positive slices")
            continue

        lower_id = int(min(positive_slices))
        upper_id = int(max(positive_slices))
        if upper_id <= lower_id:
            print(f"[WARN] Skip {pdir.name}: invalid bounds upper={upper_id}, lower={lower_id}")
            continue

        middle_candidates = [z for z in positive_slices if lower_id < z < upper_id]

        middle_rule_id = choose_middle_rule(middle_candidates)
        maxarea_rule_id = choose_maxarea_rule(gt_zyx, middle_candidates)
        maxchange_rule_id = choose_maxchange_rule(gt_zyx, middle_candidates)

        print(
            f"[INFO] {patient_id} | upper={upper_id}, lower={lower_id}, "
            f"mid={middle_rule_id}, maxarea={maxarea_rule_id}, maxchange={maxchange_rule_id}"
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            pred_mid = infer_with_upper_lower_middle(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                middle_id=middle_rule_id,
            )
            pred_maxarea = infer_with_upper_lower_middle(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                middle_id=maxarea_rule_id,
            )
            pred_maxchange = infer_with_upper_lower_middle(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                middle_id=maxchange_rule_id,
            )

            dice_mid = dice_3d(pred_mid, gt_zyx)
            dice_maxarea = dice_3d(pred_maxarea, gt_zyx)
            dice_maxchange = dice_3d(pred_maxchange, gt_zyx)

            write_mask_like(pred_mid, img_sitk, out_mid_path)
            write_mask_like(pred_maxarea, img_sitk, out_maxarea_path)
            write_mask_like(pred_maxchange, img_sitk, out_maxchange_path)

            rows.append(
                {
                    "Patient_ID": patient_id,
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Rule_Middle_Slice_ID": middle_rule_id,
                    "Rule_MaxArea_Slice_ID": maxarea_rule_id,
                    "Rule_MaxChange_Slice_ID": maxchange_rule_id,
                    "Dice3D_Rule_Middle": dice_mid,
                    "Dice3D_Rule_MaxArea": dice_maxarea,
                    "Dice3D_Rule_MaxChange": dice_maxchange,
                }
            )

            print(
                f"[OK] {patient_id}: "
                f"Dice(mid/maxarea/maxchange)={dice_mid:.4f}/{dice_maxarea:.4f}/{dice_maxchange:.4f}"
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _pid_key(row):
        mm = re.search(r"(\d+)$", str(row["Patient_ID"]))
        return int(mm.group(1)) if mm else 10**9

    rows.sort(key=_pid_key)

    wb = Workbook()
    ws = wb.active
    ws.title = "Rule_Prompt_Summary"
    ws.append(
        [
            "Patient_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Rule_Middle_Slice_ID",
            "Rule_MaxArea_Slice_ID",
            "Rule_MaxChange_Slice_ID",
            "Dice3D_Rule_Middle",
            "Dice3D_Rule_MaxArea",
            "Dice3D_Rule_MaxChange",
        ]
    )

    for r in rows:
        ws.append(
            [
                r["Patient_ID"],
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                r["Rule_Middle_Slice_ID"],
                r["Rule_MaxArea_Slice_ID"],
                r["Rule_MaxChange_Slice_ID"],
                round(float(r["Dice3D_Rule_Middle"]), 6),
                round(float(r["Dice3D_Rule_MaxArea"]), 6),
                round(float(r["Dice3D_Rule_MaxChange"]), 6),
            ]
        )

    wb.save(str(OUT_XLSX))
    print(f"[DONE] Excel saved: {OUT_XLSX}")
    print(f"[DONE] rule_middle masks: {MID_MASK_DIR}")
    print(f"[DONE] rule_maxarea masks: {MAXAREA_MASK_DIR}")
    print(f"[DONE] rule_maxchange masks: {MAXCHANGE_MASK_DIR}")


if __name__ == "__main__":
    main()
