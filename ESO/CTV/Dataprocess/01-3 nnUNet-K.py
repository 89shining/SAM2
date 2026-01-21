#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟SAM2的分段统计nnUNet的精度
Use SAM PromptSlices definition to evaluate nnUNet results.
Output patient-level CSV with SAME FORMAT as SAM oracle table.
"""

import re
import ast
import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy import metric
from pathlib import Path


# ======================================================
# 配置区（只改这里）
# ======================================================

# ① SAM 的 patient-level 统计表（你前一段代码生成的）
SAM_ORACLE_CSV = r"/home/wusi/sam2/SAM2data/20260108/Statistics/testdata/oracle_patient_level.csv"

# ② nnUNet 预测结果目录（每个患者一个 nii.gz）
NNUNET_PRED_DIR = Path(
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset008_EsoCTV73p/nnUNetTrainer__nnUNetPlans__3d_fullres/testResult_fold1"
)

# ③ GT 目录（与 nnUNet 预测一一对应）
GT_DIR = Path(
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset008_EsoCTV73p/labelsTs"
)

# ④ 输出 CSV
OUT_CSV = r"/home/wusi/sam2/SAM2data/20260108/Statistics/testdata/nnUNet_patient_level.csv"


# ======================================================
# 工具函数
# ======================================================

def load_nii(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return arr.astype(np.uint8)


def dice_3d(a, b):
    return metric.binary.dc(a, b)


def hd95_3d(a, b, spacing=(1, 1, 1)):
    if np.sum(a) == 0 or np.sum(b) == 0:
        return np.nan
    return metric.binary.hd95(a, b, voxelspacing=spacing)


def parse_prompt_slices(s):
    """
    "[0, 43]" -> set([0, 43])
    """
    if pd.isna(s):
        return set()
    return set(ast.literal_eval(s))


# ======================================================
# 主逻辑
# ======================================================

def main():
    sam_df = pd.read_csv(SAM_ORACLE_CSV)

    rows = []

    for _, row in sam_df.iterrows():
        pid = row["PatientID"]
        K = int(row["K"])
        prompt_slices = parse_prompt_slices(row["PromptSlices"])

        # ---------- 文件名匹配 ----------
        # 例：p_10 -> CTV_010.nii.gz（按你之前的规则）
        pid_num = int(re.search(r"\d+", pid).group())
        fname = f"CTV_{pid_num:03d}.nii.gz"

        pred_path = NNUNET_PRED_DIR / fname
        gt_path = GT_DIR / fname

        if not pred_path.exists() or not gt_path.exists():
            print(f"[WARN] Missing file for {pid}, skip")
            continue

        pred = load_nii(pred_path)
        gt = load_nii(gt_path)

        assert pred.shape == gt.shape

        # ---------- All ----------
        dice_all = dice_3d(pred, gt)
        hd_all = hd95_3d(pred, gt)

        # ---------- NoPrompt ----------
        mask_np = np.ones(pred.shape[0], dtype=bool)
        for z in prompt_slices:
            if 0 <= z < pred.shape[0]:
                mask_np[z] = False

        pred_np = pred[mask_np]
        gt_np = gt[mask_np]

        if np.sum(gt_np) == 0:
            dice_np = np.nan
            hd_np = np.nan
        else:
            dice_np = dice_3d(pred_np, gt_np)
            hd_np = hd95_3d(pred_np, gt_np)

        rows.append({
            "PatientID": pid,
            "K": K,
            "Dice3D_All": round(dice_all, 4),
            "Dice3D_NoPrompt": round(dice_np, 4) if not np.isnan(dice_np) else np.nan,
            "HD95_All": round(hd_all, 4) if not np.isnan(hd_all) else np.nan,
            "HD95_NoPrompt": round(hd_np, 4) if not np.isnan(hd_np) else np.nan,
            "PromptSlices": row["PromptSlices"],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] nnUNet patient-level CSV saved to:\n{OUT_CSV}")


if __name__ == "__main__":
    main()
