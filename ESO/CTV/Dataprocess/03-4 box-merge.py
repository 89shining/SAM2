"""
对新增的每张切片每个指标
按 stage(由K合并) 分组，在同一个横坐标下并排画多个指标的箱线图
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
# 3. 四个指标（用你真实列名）
# =========================
# 注：你的表里不是 area_change_percentile/center_shift_percentile，而是 rank列
metrics = [
    ("area_percentile", "Area Percentile"),
    ("delta_area_rank", "Area Change Percentile"),
    ("delta_center_rank", "Center Shift Percentile"),
    ("z_rel", "Relative Z Position"),
]

# =========================
# 4. 构造分组并排箱线图数据
# =========================
box_data = []
positions = []
tick_positions = []  # 每个 stage 的中心位置，用来放 stage 的刻度标签

group_gap = 2.0      # 不同stage之间的间距（可调大一点更清楚）
box_width = 0.6

pos = 1.0
n_metrics = len(metrics)

for stage in stage_order:
    start_pos = pos
    for i, (col, _) in enumerate(metrics):
        values = df[df["stage"] == stage][col].dropna().values
        box_data.append(values)
        positions.append(pos + i)
    end_pos = pos + (n_metrics - 1)
    tick_positions.append((start_pos + end_pos) / 2.0)
    pos = end_pos + group_gap + 1.0  # 下一组起点

# =========================
# 5. 绘图
# =========================
plt.figure(figsize=(12, 5))

plt.boxplot(
    box_data,
    positions=positions,
    widths=box_width,
    showfliers=True
)

# stage 作为主刻度
plt.xticks(tick_positions, stage_order)
plt.xlabel("K Group (Merged)")
plt.ylabel("Percentile / Rank (0–1)")
plt.title("Grouped Boxplot of Slice-level Metrics Across K Groups")

# 用第二层刻度标出每个箱子对应哪个指标（可选但很有用）
# 我们在每个stage下重复标一次指标名
minor_tick_positions = []
minor_tick_labels = []
pos = 1.0
for _stage in stage_order:
    for i, (_col, short_name) in enumerate(metrics):
        minor_tick_positions.append(pos + i)
        minor_tick_labels.append(short_name)
    pos = (pos + (n_metrics - 1)) + group_gap + 1.0

ax = plt.gca()
ax.set_xticks(minor_tick_positions, minor=True)
ax.set_xticklabels(minor_tick_labels, minor=True, rotation=0, fontsize=9)

# 让主刻度和次刻度不要挤在一起
ax.tick_params(axis="x", which="major", pad=18)
ax.tick_params(axis="x", which="minor", pad=2)

plt.tight_layout()

out_png = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\boxplot\boxplot_grouped_merged_metrics.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Saved: {out_png}")
