"""
将nii.gz数据转为mose数据集格式
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert medical NIfTI dataset to SAM2 VOS format.

Input:
  SRC_ROOT/
    p_xxx/
      image.nii.gz
      CTV.nii.gz

Output:
  OUT_ROOT/
    JPEGImages/p_xxx/00000.jpg ...
    Annotations/p_xxx/00000.png ...
    ImageSets/train.txt
"""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm


# =========================
# ======= CONFIG ==========
# =========================

SRC_ROOT = Path("/home/wusi/SAMdata/Eso/20251217_CTV/cropnii_nnUNet/train_nii")
OUT_ROOT = Path("/home/wusi/sam2/SAM2data/20260108/TrainData")

IMAGE_NAME = "image.nii.gz"
MASK_NAME  = "CTV.nii.gz"

# CT window / level
WINDOW_CENTER = 40
WINDOW_WIDTH  = 400

# Whether to only keep GT z-span
KEEP_GT_SPAN = True


# =========================
# ======= UTILS ===========
# =========================

def window_level(ct: np.ndarray, wc: float, ww: float) -> np.ndarray:
    ct = ct.astype(np.float32)
    lo = wc - ww / 2
    hi = wc + ww / 2
    ct = np.clip(ct, lo, hi)
    ct = (ct - lo) / (hi - lo + 1e-6)
    return (ct * 255).astype(np.uint8)


def find_gt_span(mask_zyx: np.ndarray):
    z_nonzero = np.where(mask_zyx.reshape(mask_zyx.shape[0], -1).sum(1) > 0)[0]
    if len(z_nonzero) == 0:
        return 0, mask_zyx.shape[0] - 1
    return int(z_nonzero[0]), int(z_nonzero[-1])


# =========================
# ======= MAIN ============
# =========================

def main():

    jpeg_root = OUT_ROOT / "JPEGImages"
    ann_root  = OUT_ROOT / "Annotations"
    split_dir = OUT_ROOT / "ImageSets"

    jpeg_root.mkdir(parents=True, exist_ok=True)
    ann_root.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    case_names = sorted([p.name for p in SRC_ROOT.iterdir() if p.is_dir()])

    train_list = []

    for case in tqdm(case_names, desc="Processing cases"):
        case_dir = SRC_ROOT / case

        img_nii = case_dir / IMAGE_NAME
        msk_nii = case_dir / MASK_NAME

        if not img_nii.exists() or not msk_nii.exists():
            print(f"[Skip] Missing files in {case}")
            continue

        # Read NIfTI
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_nii)))  # (Z,Y,X)
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_nii)))  # (Z,Y,X)

        msk = (msk > 0).astype(np.uint8)

        if KEEP_GT_SPAN:
            z0, z1 = find_gt_span(msk)
        else:
            z0, z1 = 0, img.shape[0] - 1

        out_img_dir = jpeg_root / case
        out_msk_dir = ann_root / case
        out_img_dir.mkdir(exist_ok=True)
        out_msk_dir.mkdir(exist_ok=True)

        frame_idx = 0
        for z in range(z0, z1 + 1):
            ct_slice = img[z]
            mask_slice = msk[z]

            # ---- CT → JPEG (RGB) ----
            ct_wl = window_level(ct_slice, WINDOW_CENTER, WINDOW_WIDTH)
            ct_rgb = np.stack([ct_wl] * 3, axis=-1)  # (H,W,3)

            img_name = f"{frame_idx:05d}.jpg"
            cv2.imwrite(
                str(out_img_dir / img_name),
                cv2.cvtColor(ct_rgb, cv2.COLOR_RGB2BGR),
            )

            # ---- Mask → PNG (single channel, 0/1) ----
            mask_name = f"{frame_idx:05d}.png"
            cv2.imwrite(
                str(out_msk_dir / mask_name),
                mask_slice.astype(np.uint8),
            )

            frame_idx += 1

        if frame_idx > 0:
            train_list.append(case)

    # Write train.txt
    with open(split_dir / "train.txt", "w") as f:
        for name in train_list:
            f.write(name + "\n")

    print("Done.")
    print(f"Total cases: {len(train_list)}")
    print(f"Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
