"""
绘制指标散点图,（K=3~10）
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ======================
# 路径配置
# ======================
slice_all_csv = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\slice_level_metrics.csv"
prompt_xlsx   = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\Prompt_incremental_analysis.xlsx"

out_dir = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\SingleK"
os.makedirs(out_dir, exist_ok=True)

# ======================
# 读取 All-slice 数据（只读一次）
# ======================
df_all = pd.read_csv(slice_all_csv)

# ======================
# 6 个特征组合（顺序保持不变）
# ======================
pairs = [
    ("z_rel", "area_percentile"),             # A
    ("z_rel", "delta_area_rank"),              # B
    ("z_rel", "delta_center_rank"),            # C
    ("delta_area_rank", "delta_center_rank"),  # F
    ("area_percentile", "delta_area_rank"),    # D
    ("area_percentile", "delta_center_rank"),  # E
]

titles = [
    "(A) z_rel vs area_percentile",
    "(B) z_rel vs delta_area_rank",
    "(C) z_rel vs delta_center_rank",
    "(F) delta_area_rank vs delta_center_rank",
    "(D) area_percentile vs delta_area_rank",
    "(E) area_percentile vs delta_center_rank",
]

# ======================
# 批量绘制 K = 3 ~ 10
# ======================
for K in range(3, 11):

    print(f"Plotting K = {K} ...")

    # 读取 Prompt slices
    df_prompt = pd.read_excel(prompt_xlsx, sheet_name=f"K{K}")

    out_fig = os.path.join(out_dir, f"K{K}_singleslice_scatter_6plots.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (x, y), title in zip(axes, pairs, titles):

        # 灰点：All slices
        ax.scatter(
            df_all[x], df_all[y],
            s=12, alpha=0.3, color="gray", label="All slices"
        )

        # 红点：Prompt slices
        ax.scatter(
            df_prompt[x], df_prompt[y],
            s=12, alpha=0.9, color="red", label=f"Prompt slices (K={K})"
        )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

    # 只放一个 legend
    axes[0].legend()

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)   # ⭐非常重要：批量时防止内存/句柄爆掉

print("✅ All K plots saved.")
