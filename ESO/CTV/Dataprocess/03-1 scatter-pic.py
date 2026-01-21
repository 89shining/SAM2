"""
绘制指标散点图,每个K的总体数据
"""

import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 路径配置
# ======================
slice_all_csv = r"C:\Users\dell\Desktop\testdata\slice_level_metrics.csv"
prompt_xlsx   = r"C:\Users\dell\Desktop\testdata\Prompt_incremental_analysis.xlsx"

K = 10
out_fig = rf"C:\Users\dell\Desktop\testdata\K{K}_singleslice_scatter_6plots.png"


# ======================
# 读取数据
# ======================
df_all = pd.read_csv(slice_all_csv)
df_prompt = pd.read_excel(prompt_xlsx, sheet_name=f"K{K}")

# ======================
# 6 个特征组合（调整顺序）
# ======================
pairs = [
    # 第一行
    ("z_rel", "area_percentile"),          # A
    ("z_rel", "delta_area_rank"),           # B
    ("z_rel", "delta_center_rank"),         # C

    # 第二行
    ("delta_area_rank", "delta_center_rank"),  # F（提前）
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
# 画图
# ======================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (x, y), title in zip(axes, pairs, titles):

    # 灰点：All slices
    ax.scatter(
        df_all[x], df_all[y],
        s=12, alpha=0.3, color="gray", label="All slices"
    )

    # 红点：Prompt slices (K=3)
    ax.scatter(
        df_prompt[x], df_prompt[y],
        s=12, alpha=0.9, color="red", label="Prompt slices"
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

# 只放一个 legend
axes[0].legend()

plt.tight_layout()
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.show()
