"""
计算每个指标和K的相关性
以及和分组K的相关性
"""

import pandas as pd
from scipy.stats import spearmanr

# =========================
# 1. 读取数据
# =========================
xlsx_path = r"C:\Users\dell\Desktop\testdata\Prompt_incremental_analysis.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="Incremental_All")

# =========================
# 2. 需要分析的指标
# =========================
metrics = {
    "area_percentile": "Area Percentile",
    "delta_area_rank": "Area Change Percentile",
    "delta_center_rank": "Center Shift Percentile",
    "z_rel": "Relative Z Position",
}

print("=== Spearman correlation: metric vs K ===")

for col, name in metrics.items():
    rho, p = spearmanr(df["K"], df[col])
    print(f"{name:25s}  rho = {rho:6.3f},  p = {p:.3e}")

# =========================
# 3. K → stage index
# =========================
def k_to_stage_idx(k):
    if k in [3, 4]:
        return 0   # early
    elif k in [5, 6, 7]:
        return 1   # mid
    elif k in [8, 9, 10]:
        return 2   # late
    else:
        return None

df["stage_idx"] = df["K"].apply(k_to_stage_idx)
df = df.dropna(subset=["stage_idx"])

print("\n=== Spearman correlation: metric vs stage index ===")

for col, name in metrics.items():
    rho, p = spearmanr(df["stage_idx"], df[col])
    print(f"{name:25s}  rho = {rho:6.3f},  p = {p:.3e}")
