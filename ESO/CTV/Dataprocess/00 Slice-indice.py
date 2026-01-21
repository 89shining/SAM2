"""
统计每个患者每张切片的指标：z位置，面积，面积变化率，中心位移
"""

import os
import re

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass


def compute_slice_metrics_from_nii(
    nii_path,
    patient_id,
    round_decimals=2,
):
    """
    从单个 CTV_xxx.nii.gz 计算 slice-level 指标
    返回 pandas DataFrame
    """

    # ======================
    # 1. 读取 NIfTI
    # ======================
    nii = nib.load(nii_path)
    mask = nii.get_fdata() > 0  # binary
    spacing = nii.header.get_zooms()[:3]  # (x, y, z)
    pixel_area = spacing[0] * spacing[1]

    Z = mask.shape[2]

    # ======================
    # 2. 找 GT 非空 slice
    # ======================
    slice_indices = [z for z in range(Z) if mask[:, :, z].any()]
    if len(slice_indices) < 3:
        return None  # 不足以定义中间 slice

    z_min, z_max = slice_indices[0], slice_indices[-1]

    # 中间 slice（去掉上下界）
    middle_slices = slice_indices[1:-1]

    # ======================
    # 3. 逐 slice 计算原始量
    # ======================
    areas = {}
    centers = {}

    for z in slice_indices:
        slice_mask = mask[:, :, z]
        areas[z] = slice_mask.sum() * pixel_area
        centers[z] = center_of_mass(slice_mask)

    rows = []

    for z in middle_slices:
        # ---- 面积变化（raw）----
        delta_area_raw = abs(areas[z] - areas[z - 1]) / areas[z - 1]

        # ---- 中心点变化（raw）----
        c0 = np.array(centers[z - 1])
        c1 = np.array(centers[z])
        delta_center_raw = np.linalg.norm(c1 - c0)

        rows.append({
            "patient_id": patient_id,
            "z": z,
            "area": areas[z],
            "delta_area_raw": delta_area_raw,
            "delta_center_raw": delta_center_raw,
        })

    df = pd.DataFrame(rows)

    # ======================
    # 4. z_rel
    # ======================
    df["z_rel"] = (df["z"] - z_min) / (z_max - z_min)

    # ======================
    # 5. area_percentile（用所有 GT 非空 slice）
    # ======================
    all_areas = np.array([areas[z] for z in slice_indices])
    sorted_areas = np.sort(all_areas)

    def area_percentile(a):
        rank = np.where(sorted_areas == a)[0][-1]
        return rank / (len(sorted_areas) - 1)

    df["area_percentile"] = df["area"].apply(area_percentile)

    # ======================
    # 6. rank（患者内排序）
    # ======================
    def rank_01(x):
        order = x.rank(method="min")
        return (order - 1) / (len(x) - 1)

    df["delta_area_rank"] = rank_01(df["delta_area_raw"])
    df["delta_center_rank"] = rank_01(df["delta_center_raw"])

    # ======================
    # 7. 保留两位小数
    # ======================
    for col in [
        "area",
        "z_rel",
        "area_percentile",
        "delta_area_raw",
        "delta_area_rank",
        "delta_center_raw",
        "delta_center_rank",
    ]:
        df[col] = df[col].round(round_decimals)

    # 列顺序整理
    df = df[
        [
            "patient_id",
            "z",
            "z_rel",
            "area",
            "area_percentile",
            "delta_area_raw",
            "delta_area_rank",
            "delta_center_raw",
            "delta_center_rank",
        ]
    ]

    return df

def extract_patient_index(fname):
    """
    从 CTV_000.nii.gz 中提取数字 0
    """
    m = re.search(r"(\d+)", fname)
    if m is None:
        raise ValueError(f"Cannot extract patient index from {fname}")
    return int(m.group(1))


def process_folder(
    nii_dir,
    out_csv,
):
    all_rows = []

    # =========================
    # 1. 按“数字编号”排序文件
    # =========================
    nii_files = [
        f for f in os.listdir(nii_dir)
        if f.endswith(".nii.gz")
    ]
    nii_files = sorted(nii_files, key=extract_patient_index)

    for fname in nii_files:
        idx = extract_patient_index(fname)

        # 统一 patient_id 格式
        patient_id = f"p_{idx}"

        nii_path = os.path.join(nii_dir, fname)

        df = compute_slice_metrics_from_nii(
            nii_path=nii_path,
            patient_id=patient_id,
            round_decimals=2,
        )

        if df is not None:
            all_rows.append(df)

    if len(all_rows) == 0:
        raise RuntimeError("No valid patients found")

    df_all = pd.concat(all_rows, ignore_index=True)

    # 再保险一次：按 patient_id 数字 + z 排序
    df_all["patient_idx"] = df_all["patient_id"].str.replace("p_", "", regex=False).astype(int)
    df_all = df_all.sort_values(["patient_idx", "z"])
    df_all = df_all.drop(columns=["patient_idx"])

    df_all.to_csv(out_csv, index=False)
    print(f"[OK] Saved slice-level metrics to: {out_csv}")


# ========= 使用示例 =========
if __name__ == "__main__":
    nii_dir = r"C:\Users\dell\Desktop\Eso-CTV\TestResult\TestResult\labelsTs"
    out_csv = r"C:\Users\dell\Desktop/testdata/slice_level_metrics.csv"

    process_folder(nii_dir, out_csv)

