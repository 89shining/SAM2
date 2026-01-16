#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计汇总成 Excel 表（Summary 为 均值±标准差）
绘制 K-Dice / K-HD95 曲线（分开保存）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ======================================================
# 配置区（只改这里）
# ======================================================

ORACLE_CSV = r"/home/wusi/sam2/SAM2data/20260108/Statistics/test-try/oracle_patient_level.csv"
OUT_XLSX   = r"/home/wusi/sam2/SAM2data/20260108/Statistics/test-try/Oracle_Summary.xlsx"

OUT_DICE_FIG = r"/home/wusi/sam2/SAM2data/20260108/Statistics/test-try/K_Dice.png"
OUT_HD95_FIG = r"/home/wusi/sam2/SAM2data/20260108/Statistics/test-try/K_HD95.png"

K_RANGE = list(range(2, 11))
DECIMALS = 2


# ======================================================
# 工具函数
# ======================================================

def mean_std_str(mean, std, decimals=2):
    """格式化为 均值±标准差"""
    if np.isnan(mean) or np.isnan(std):
        return "-"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


# ======================================================
# 主逻辑
# ======================================================

def main():
    df = pd.read_csv(ORACLE_CSV)

    # ----------- 基本检查 -----------
    required = {
        "PatientID",
        "K",
        "Dice3D_All",
        "Dice3D_NoPrompt",
        "HD95_All",
        "HD95_NoPrompt",
        "PromptSlices",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["K"] = df["K"].astype(int)

    # ==================================================
    # 1) Summary（均值±标准差）
    # ==================================================
    summary_rows = []

    for K in K_RANGE:
        df_k = df[df["K"] == K]

        if len(df_k) == 0:
            row = {
                "K": K,
                "Dice_All": "-",
                "HD95_All": "-",
                "Dice_NoPrompt": "-",
                "HD95_NoPrompt": "-",
            }
        else:
            row = {
                "K": K,
                "Dice_All": mean_std_str(
                    df_k["Dice3D_All"].mean(),
                    df_k["Dice3D_All"].std(),
                    DECIMALS,
                ),
                "HD95_All": mean_std_str(
                    df_k["HD95_All"].mean(),
                    df_k["HD95_All"].std(),
                    DECIMALS,
                ),
                "Dice_NoPrompt": mean_std_str(
                    df_k["Dice3D_NoPrompt"].mean(),
                    df_k["Dice3D_NoPrompt"].std(),
                    DECIMALS,
                ),
                "HD95_NoPrompt": mean_std_str(
                    df_k["HD95_NoPrompt"].mean(),
                    df_k["HD95_NoPrompt"].std(),
                    DECIMALS,
                ),
            }

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary[
        ["K", "Dice_All", "HD95_All", "Dice_NoPrompt", "HD95_NoPrompt"]
    ]

    # ==================================================
    # 2) 写 Excel
    # ==================================================
    out_path = Path(OUT_XLSX)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Summary
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # K2 ... K10
        for K in K_RANGE:
            df_k = df[df["K"] == K].copy()
            if len(df_k) == 0:
                continue

            df_k = df_k.sort_values("PatientID").reset_index(drop=True)
            df_k.to_excel(writer, sheet_name=f"K{K}", index=False)

    print(f"[OK] Excel saved to: {OUT_XLSX}")

    # ==================================================
    # 3) 画 Dice 曲线
    # ==================================================
    ks = np.array(K_RANGE)

    dice_all_mean = [
        df[df["K"] == K]["Dice3D_All"].mean() for K in K_RANGE
    ]
    dice_all_std = [
        df[df["K"] == K]["Dice3D_All"].std() for K in K_RANGE
    ]

    dice_np_mean = [
        df[df["K"] == K]["Dice3D_NoPrompt"].mean() for K in K_RANGE
    ]
    dice_np_std = [
        df[df["K"] == K]["Dice3D_NoPrompt"].std() for K in K_RANGE
    ]

    plt.figure(figsize=(6.5, 4.5))

    plt.plot(ks, dice_all_mean, marker="o", linewidth=2, label="Dice3D_All")
    plt.fill_between(
        ks,
        np.array(dice_all_mean) - np.array(dice_all_std),
        np.array(dice_all_mean) + np.array(dice_all_std),
        alpha=0.25,
    )

    plt.plot(
        ks,
        dice_np_mean,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Dice3D_NoPrompt",
    )
    plt.fill_between(
        ks,
        np.array(dice_np_mean) - np.array(dice_np_std),
        np.array(dice_np_mean) + np.array(dice_np_std),
        alpha=0.25,
    )

    plt.xlabel("Number of Prompt Slices (K)")
    plt.ylabel("Dice (3D)")
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DICE_FIG, dpi=300)
    plt.close()

    print(f"[OK] Dice curve saved to: {OUT_DICE_FIG}")

    # ==================================================
    # 4) 画 HD95 曲线
    # ==================================================
    hd_all_mean = [
        df[df["K"] == K]["HD95_All"].mean() for K in K_RANGE
    ]
    hd_all_std = [
        df[df["K"] == K]["HD95_All"].std() for K in K_RANGE
    ]

    hd_np_mean = [
        df[df["K"] == K]["HD95_NoPrompt"].mean() for K in K_RANGE
    ]
    hd_np_std = [
        df[df["K"] == K]["HD95_NoPrompt"].std() for K in K_RANGE
    ]

    plt.figure(figsize=(6.5, 4.5))

    plt.plot(ks, hd_all_mean, marker="o", linewidth=2, label="HD95_All")
    plt.fill_between(
        ks,
        np.array(hd_all_mean) - np.array(hd_all_std),
        np.array(hd_all_mean) + np.array(hd_all_std),
        alpha=0.25,
    )

    plt.plot(
        ks,
        hd_np_mean,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="HD95_NoPrompt",
    )
    plt.fill_between(
        ks,
        np.array(hd_np_mean) - np.array(hd_np_std),
        np.array(hd_np_mean) + np.array(hd_np_std),
        alpha=0.25,
    )

    plt.xlabel("Number of Prompt Slices (K)")
    plt.ylabel("HD95 (mm)")
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HD95_FIG, dpi=300)
    plt.close()

    print(f"[OK] HD95 curve saved to: {OUT_HD95_FIG}")


# ======================================================
# 入口
# ======================================================

if __name__ == "__main__":
    main()
