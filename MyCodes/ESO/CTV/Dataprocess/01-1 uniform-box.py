#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2 inference (box prompt) + evaluation
K = 2..10

Output:
  oracle_patient_level.csv

⚠️ 不保存任何预测 NIfTI，仅用于评估
"""

import os
import sys
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import torch
from medpy.metric import binary as medpy_binary

# ======================================================
# SAM2 import
# ======================================================
sys.path.append("/home/wusi/sam2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sam2.build_sam import build_sam2_video_predictor


# ======================================================
# 路径 & 配置（只改这里）
# ======================================================
DATA_ROOT = Path(
    "/home/wusi/sam2/SAM2data/20260108/Data83"
)

OUT_CSV = Path(
    "/home/wusi/sam2/SAM2data/20260108/Statistics/AAPM/Uniform_Box_oracle_patient_level.csv"
)

IMG_NAME = "image.nii.gz"
GT_NAME  = "CTV.nii.gz"

SAM2_CKPT = Path(
    "/home/wusi/sam2/checkpoints/sam2.1_hiera_large.pt"
)
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"

WINDOW_CENTER = 40
WINDOW_WIDTH  = 400

K_START = 2
K_END   = 10

OBJ_ID = 1


# ======================================================
# Utils
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


def choose_prompt_slices(z_len, K):
    idx = np.linspace(0, z_len - 1, K).round().astype(int)
    return sorted(set(idx.tolist()))


def read_nii_zyx(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def mask_to_box(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def exclude_prompt_slices(vol, prompt_slices):
    vol = vol.copy()
    for z in prompt_slices:
        if 0 <= z < vol.shape[0]:
            vol[z] = 0
    return vol


# ======================================================
# SAM2 inference (box prompt)
# ======================================================
@torch.no_grad()
def sam2_infer_one_patient(predictor, img_zyx, gt_zyx, prompt_slices, frame_dir):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    for s in prompt_slices:
        mask = (gt_zyx[s] > 0).astype(np.uint8)
        box = mask_to_box(mask)
        if box is None:
            continue

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=int(s),
            obj_id=OBJ_ID,
            box=np.array(box, dtype=np.float32),
        )

    z, h, w = img_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)

    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == OBJ_ID:
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break

    return pred


# ======================================================
# Main
# ======================================================
def main():
    rows = []

    device = torch.device(DEVICE)
    predictor = build_sam2_video_predictor(
        SAM2_CFG, str(SAM2_CKPT), device=device
    )

    patient_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"[INFO] Found {len(patient_dirs)} patients")

    for K in range(K_START, K_END + 1):
        print(f"\n===== K = {K} =====")

        for pdir in patient_dirs:
            m = re.fullmatch(r"p_(\d+)", pdir.name)
            if not m:
                continue

            pid = int(m.group(1))
            pid_str = f"p_{pid}"

            img_zyx, _ = read_nii_zyx(pdir / IMG_NAME)
            gt_zyx, gt_itk = read_nii_zyx(pdir / GT_NAME)

            prompt_slices = choose_prompt_slices(img_zyx.shape[0], K)

            tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pid_str}_K{K}_"))
            try:
                save_frames_from_volume(img_zyx, tmp_dir)
                pred = sam2_infer_one_patient(
                    predictor, img_zyx, gt_zyx, prompt_slices, tmp_dir
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # ========= metrics =========
            spacing = gt_itk.GetSpacing()[::-1]  # (z, y, x)

            dice_all = medpy_binary.dc(pred, gt_zyx)
            hd95_all = medpy_binary.hd95(pred, gt_zyx, voxelspacing=spacing)

            pred_np = exclude_prompt_slices(pred, prompt_slices)
            gt_np   = exclude_prompt_slices(gt_zyx, prompt_slices)

            dice_np = medpy_binary.dc(pred_np, gt_np)
            hd95_np = medpy_binary.hd95(pred_np, gt_np, voxelspacing=spacing)

            rows.append({
                "PatientID": pid_str,
                "K": K,
                "Dice3D_All": round(dice_all, 2),
                "Dice3D_NoPrompt": round(dice_np, 2),
                "HD95_All": round(hd95_all, 2),
                "HD95_NoPrompt": round(hd95_np, 2),
                "PromptSlices": str(prompt_slices),
            })

    df = pd.DataFrame(rows)

    # 排序：患者号 → K
    df["_pid"] = df["PatientID"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(by=["_pid", "K"]).drop(columns="_pid")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Final CSV saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
