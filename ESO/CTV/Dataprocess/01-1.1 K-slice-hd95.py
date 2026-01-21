#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HD95 + Dice constrained Oracle builder

- K=2: 固定上下界
- K>=3: 在 Dice 不明显下降的前提下，最小化 HD95（均排除提示层）

Output:
oracle_patient_level.csv
"""

import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import torch
from medpy.metric import binary as medpy_binary

sys.path.append("/home/wusi/sam2")
from sam2.build_sam import build_sam2_video_predictor


# ===================== 配置区 =====================

DATA_ROOT = Path("/home/wusi/SAMdata/Eso/20260104_CTV/nnUNet_mask/cropdatanii/test_nii")
IMG_NAME = "image.nii.gz"
GT_NAME  = "CTV.nii.gz"

SAM2_CKPT = "/home/wusi/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"

WINDOW_CENTER = 40
WINDOW_WIDTH  = 400

K_MAX = 10
OBJ_ID = 1

OUT_CSV = "/home/wusi/sam2/SAM2data/20260108/Statistics/testdata/oracle_patient_level_hd95_constrained.csv"

ROUND_N = 2

# ===== Oracle 约束参数 =====
DICE_TOL = 0.98          # Dice 允许最多下降 2%
HD95_INF = 1e6


# ===================== 工具函数 =====================

def window_to_uint8(img2d, wc, ww):
    lo, hi = wc - ww / 2, wc + ww / 2
    img = np.clip(img2d, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255
    return img.astype(np.uint8)

def save_frames(vol, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(vol.shape[0]):
        u8 = window_to_uint8(vol[i], WINDOW_CENTER, WINDOW_WIDTH)
        rgb = np.stack([u8]*3, axis=-1)
        Image.fromarray(rgb).save(out_dir / f"{i:05d}.jpg")

def read_nii(path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img

def dice_3d(a, b, eps=1e-5):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return (2 * inter + eps) / (s + eps) if s > 0 else 1.0

def hd95_3d(a, b, spacing=None):
    a = a.astype(bool)
    b = b.astype(bool)

    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return np.nan

    return medpy_binary.hd95(a, b, voxelspacing=spacing)

def get_z_bounds(gt):
    z = np.where(gt.reshape(gt.shape[0], -1).sum(axis=1) > 0)[0]
    return int(z.min()), int(z.max())

@torch.no_grad()
def sam2_infer(predictor, gt, prompt_slices, frame_dir):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    for z in prompt_slices:
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(z),
            obj_id=OBJ_ID,
            mask=(gt[z] > 0).astype(np.uint8),
        )

    pred = np.zeros_like(gt, dtype=np.uint8)
    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == OBJ_ID:
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break
    return pred

def dice_excl_prompt(pred, gt, prompt_slices):
    mask = np.ones(gt.shape[0], dtype=bool)
    mask[list(prompt_slices)] = False
    return dice_3d(pred[mask], gt[mask])

def hd95_excl_prompt(pred, gt, prompt_slices, spacing=None):
    mask = np.ones(gt.shape[0], dtype=bool)
    mask[list(prompt_slices)] = False
    return hd95_3d(pred[mask], gt[mask], spacing)


# ===================== 主流程 =====================

def main():
    predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
    results = []

    for pdir in sorted(DATA_ROOT.iterdir()):
        if not pdir.is_dir():
            continue

        pid = pdir.name
        img, _ = read_nii(pdir / IMG_NAME)
        gt, gt_itk = read_nii(pdir / GT_NAME)

        spacing = gt_itk.GetSpacing()[::-1]  # (z, y, x)
        gt = (gt > 0).astype(np.uint8)

        z_low, z_high = get_z_bounds(gt)
        mid_candidates = list(range(z_low + 1, z_high))

        tmp = Path(tempfile.mkdtemp())
        try:
            save_frames(img, tmp)

            # ===== K = 2（仅上下界）=====
            prompt = [z_low, z_high]
            pred = sam2_infer(predictor, gt, prompt, tmp)

            results.append({
                "PatientID": pid,
                "K": 2,
                "Dice3D_All": round(dice_3d(pred, gt), ROUND_N),
                "Dice3D_NoPrompt": round(dice_excl_prompt(pred, gt, prompt), ROUND_N),
                "HD95_All": round(hd95_3d(pred, gt, spacing), ROUND_N),
                "HD95_NoPrompt": round(hd95_excl_prompt(pred, gt, prompt, spacing), ROUND_N),
                "PromptSlices": str(prompt),
            })

            best_mid = []

            # ===== K >= 3：HD95 + Dice 约束 Oracle =====
            for K in range(3, K_MAX + 1):

                best_hd = HD95_INF
                best_dice = -1
                best_z = None

                for z in mid_candidates:
                    if z in best_mid:
                        continue

                    trial = [z_low, z_high] + best_mid + [z]
                    pred_tmp = sam2_infer(predictor, gt, trial, tmp)

                    d = dice_excl_prompt(pred_tmp, gt, trial)
                    h = hd95_excl_prompt(pred_tmp, gt, trial, spacing)

                    if not np.isfinite(h) or d <= 0:
                        continue

                    if best_z is None:
                        best_hd = h
                        best_dice = d
                        best_z = z
                        continue

                    if d < best_dice * DICE_TOL:
                        continue

                    if h < best_hd:
                        best_hd = h
                        best_dice = d
                        best_z = z

                if best_z is not None:
                    best_mid.append(best_z)

                trial = [z_low, z_high] + best_mid
                pred = sam2_infer(predictor, gt, trial, tmp)

                results.append({
                    "PatientID": pid,
                    "K": K,
                    "Dice3D_All": round(dice_3d(pred, gt), ROUND_N),
                    "Dice3D_NoPrompt": round(dice_excl_prompt(pred, gt, trial), ROUND_N),
                    "HD95_All": round(hd95_3d(pred, gt, spacing), ROUND_N),
                    "HD95_NoPrompt": round(hd95_excl_prompt(pred, gt, trial, spacing), ROUND_N),
                    "PromptSlices": str(trial),
                })

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"✔ HD95 + Dice constrained oracle results saved to {out_path}")


if __name__ == "__main__":
    main()
