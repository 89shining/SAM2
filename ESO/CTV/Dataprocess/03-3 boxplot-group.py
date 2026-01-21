"""
对新增的每张切片每个指标单独绘制箱式图：4张
按K分组绘制，横坐标：K分组
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取数据
# =========================
xlsx_path = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\Prompt_incremental_analysis.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="Incremental_All")

# =========================
# 2. 定义 K → stage 的映射
# =========================
def k_to_stage(k):
    if k in [3, 4]:
        return "early (K=3–4)"
    elif k in [5, 6, 7]:
        return "mid (K=5–7)"
    elif k in [8, 9, 10]:
        return "late (K=8–10)"
    else:
        return None

df["stage"] = df["K"].apply(k_to_stage)
df = df.dropna(subset=["stage"])

stage_order = ["early (K=3–4)", "mid (K=5–7)", "late (K=8–10)"]

# =========================
# 3. 四个指标
# =========================
metrics = {
    "area_percentile": "Area Percentile",
    "delta_area_rank": "Area Change Percentile",
    "delta_center_rank": "Center Shift Percentile",
    "z_rel": "Relative Z Position",
}

# =========================
# 4. 逐个指标画箱线图
# =========================
for metric, ylabel in metrics.items():
    data_by_stage = []

    for stage in stage_order:
        values = df[df["stage"] == stage][metric].dropna().values
        data_by_stage.append(values)

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        data_by_stage,
        tick_labels=stage_order,
        showfliers=True
    )
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Merged K Groups)")
    plt.tight_layout()

    out_png = rf"C:\Users\WS\Desktop\Esophagus\AAPM投稿\boxplot\boxplot_merged_{metric}.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved: {out_png}")
