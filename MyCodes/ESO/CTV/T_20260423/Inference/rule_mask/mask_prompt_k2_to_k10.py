#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2 rule-based inference with uniform prompt-slice selection (K=2..10).

Workflow:
1) Keep only patients p_55..p_82.
2) For each patient, find GT-positive slices.
3) For each K in [2..10], uniformly select K prompt slices from GT-positive slices.
4) Run one-shot SAM2 inference with all selected prompts at once.
5) Save masks in per-K folders and export an Excel table of selected slices.
"""

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

# Keep compatible with existing environment settings.
sys.path.append("/home/wusi/SAM2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.build_sam import build_sam2_video_predictor


# ======================================================
# ================== Path & Config =====================
# ======================================================

DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260108/Data83")
OUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260423/Zero-shot/Try_rule_mask/mask_prompt")
OUT_XLSX = OUT_ROOT / "uniform_prompt_slices_k2_to_k10.xlsx"

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

SAM2_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"  # "cuda" or "cpu"
OBJ_ID = 1

K_VALUES = list(range(2, 11))
PID_MIN = 55
PID_MAX = 82

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


def gt_positive_slices(gt_zyx):
    non_empty = np.where(gt_zyx.reshape(gt_zyx.shape[0], -1).any(axis=1))[0]
    return [int(z) for z in non_empty.tolist()]


def format_slice_list(slices):
    return "[" + ", ".join(str(int(x)) for x in slices) + "]"


def uniform_select_slices(positive_slices, k):
    """
    Uniformly select up to k slices from sorted positive_slices.
    If available positive slices are fewer than k, return all positives.
    """
    s = sorted(int(x) for x in positive_slices)
    n = len(s)
    if n == 0:
        return []
    if k >= n:
        return s.copy()

    targets = np.linspace(0.0, float(n - 1), num=k)
    available = set(range(n))
    chosen_idx = []

    for t in targets:
        cand = sorted(available, key=lambda j: (abs(j - t), j))
        j = cand[0]
        chosen_idx.append(j)
        available.remove(j)

    chosen_idx = sorted(chosen_idx)
    return [s[j] for j in chosen_idx]


# ======================================================
# =================== SAM2 Inference ====================
# ======================================================


@torch.no_grad()
def infer_with_prompt_list(predictor, frame_dir, gt_zyx, prompt_ids):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    # Keep order and remove accidental duplicates.
    prompt_ids = list(dict.fromkeys(int(x) for x in prompt_ids))

    z = gt_zyx.shape[0]
    for sid in prompt_ids:
        if sid < 0 or sid >= z:
            raise RuntimeError(f"Prompt slice out of range: {sid} (z={z})")

        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")

        predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=OBJ_ID,
            mask=prompt_mask,
        )

    _, h, w = gt_zyx.shape
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
    for k in K_VALUES:
        (OUT_ROOT / f"K{k}").mkdir(parents=True, exist_ok=True)

    device = torch.device(
        DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)

    patient_dirs = []
    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"p_(\d+)", p.name)
        if m is None:
            continue
        pid_num = int(m.group(1))
        if PID_MIN <= pid_num <= PID_MAX:
            patient_dirs.append(p)

    print(
        f"[INFO] Target patients in data root: {len(patient_dirs)} "
        f"(p_{PID_MIN}..p_{PID_MAX})"
    )

    records_by_k = {k: [] for k in K_VALUES}

    for pdir in patient_dirs:
        pid_num = int(re.fullmatch(r"p_(\d+)", pdir.name).group(1))
        patient_id = f"CTV_{pid_num:03d}"

        img_path = pdir / IMG_NAME
        gt_path = pdir / GT_NAME
        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] Skip {pdir.name}: missing image or GT")
            continue

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)
        gt_zyx = (gt_zyx > 0).astype(np.uint8)

        if img_zyx.shape != gt_zyx.shape:
            print(
                f"[WARN] Skip {pdir.name}: shape mismatch img{img_zyx.shape} vs gt{gt_zyx.shape}"
            )
            continue

        positive_slices = gt_positive_slices(gt_zyx)
        if len(positive_slices) == 0:
            print(f"[WARN] Skip {pdir.name}: GT has no positive slices")
            continue

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            for k in K_VALUES:
                prompts = uniform_select_slices(positive_slices, k)
                if len(prompts) == 0:
                    print(f"[WARN] {pdir.name} K{k}: no valid prompts, skip")
                    continue

                try:
                    pred = infer_with_prompt_list(
                        predictor=predictor,
                        frame_dir=tmp_dir,
                        gt_zyx=gt_zyx,
                        prompt_ids=prompts,
                    )
                except Exception as e:
                    print(f"[WARN] {pdir.name} K{k} inference failed: {e}")
                    continue

                out_mask_path = OUT_ROOT / f"K{k}" / f"{patient_id}.nii.gz"
                write_mask_like(pred, img_sitk, out_mask_path)

                records_by_k[k].append(
                    {
                        "PatientID": pdir.name,
                        "K": k,
                        "PromptCount": len(prompts),
                        "PromptSlices": format_slice_list(prompts),
                    }
                )

                print(f"[OK] {pdir.name} K{k} -> {out_mask_path} | prompts={prompts}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Save prompt-selection records.
    wb = Workbook()

    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_summary.append(["K", "Patients", "Missing"])

    total_patients = len(patient_dirs)
    for k in K_VALUES:
        ws = wb.create_sheet(f"K{k}")
        ws.append(["PatientID", "K", "PromptCount", "PromptSlices"])

        rows = sorted(records_by_k[k], key=lambda r: int(str(r["PatientID"]).split("_")[-1]))
        for r in rows:
            ws.append([r["PatientID"], r["K"], r["PromptCount"], r["PromptSlices"]])

        ws_summary.append([k, len(rows), max(0, total_patients - len(rows))])

    wb.save(str(OUT_XLSX))
    print(f"[DONE] Prompt table saved: {OUT_XLSX}")
    print("[DONE] All requested K inference finished.")


if __name__ == "__main__":
    main()
