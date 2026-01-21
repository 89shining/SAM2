"""
对新增的每张切片每个指标单独绘制箱式图：4张
横坐标K：3-10
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取数据
# =========================
xlsx_path = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\Prompt_incremental_analysis.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="Incremental_All")

# 保证 K 是有序的
Ks = sorted(df["K"].unique())

# =========================
# 2. 定义要画的 4 个指标
# =========================
metrics = {
    "z_rel": "Relative Z Position",
    "area_percentile": "Area Percentile",
    "delta_area_rank": "Area Change Percentile",
    "delta_center_rank": "Center Shift Percentile",
}

# =========================
# 3. 逐个指标画箱式图
# =========================
for metric, ylabel in metrics.items():
    data_by_k = []

    for k in Ks:
        values = df[df["K"] == k][metric].dropna().values
        data_by_k.append(values)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_by_k, tick_labels=Ks, showfliers=True)

    plt.xlabel("K (Prompt Number)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Distribution Across K")
    plt.tight_layout()

    # 保存图片
    out_png = rf"C:\Users\WS\Desktop\Esophagus\AAPM投稿\boxplot\boxplot_{metric}.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved: {out_png}")
