"""
分析每个患者每个K的新增层slice指标
"""

import pandas as pd
import re

# =========================
# 1. 读取数据
# =========================
xlsx_path = r"C:\Users\dell\Desktop\AAPM投稿\Prompt_slice_level_metrics.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="All_Prompt")

# =========================
# 2. 提取 patient 数字编号，用于排序
# =========================
def extract_patient_num(pid):
    """
    从 p_0 / p_10 这种字符串中提取数字
    """
    m = re.search(r"\d+", str(pid))
    return int(m.group()) if m else -1

df["patient_num"] = df["patient_id"].apply(extract_patient_num)

# =========================
# 3. 排序（先 patient_num，再 K）
# =========================
df = df.sort_values(by=["patient_num", "K"])

# =========================
# 4. 找出每个 K 相比 K-1 新增的 slice
# =========================
incremental_rows = []

for patient_id, df_p in df.groupby("patient_id"):
    df_p = df_p.sort_values("K")
    Ks = sorted(df_p["K"].unique())

    for k in Ks:
        if k == Ks[0]:
            # 第一个 K，认为全部是新增
            new_rows = df_p[df_p["K"] == k]
        else:
            df_prev = df_p[df_p["K"] == k - 1]
            df_curr = df_p[df_p["K"] == k]

            prev_z = set(df_prev["z"])
            curr_z = set(df_curr["z"])

            new_z = curr_z - prev_z
            new_rows = df_curr[df_curr["z"].isin(new_z)]

        for _, row in new_rows.iterrows():
            incremental_rows.append(row)

df_incremental = pd.DataFrame(incremental_rows)

# =========================
# 5. 再次按 patient_num + K 排序（保证输出顺序）
# =========================
df_incremental["patient_num"] = df_incremental["patient_id"].apply(extract_patient_num)
df_incremental = df_incremental.sort_values(by=["patient_num", "K"])

# =========================
# 6. 保存为新的 Excel
# =========================
out_xlsx = r"C:\Users\dell\Desktop\AAPM投稿\Prompt_incremental_analysis.xlsx"

with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    # Sheet 1：全部新增 slice
    df_incremental.drop(columns=["patient_num"]).to_excel(
        writer, sheet_name="Incremental_All", index=False
    )

    # 后续：按 K 单独一个 sheet
    for k in sorted(df_incremental["K"].unique()):
        df_k = df_incremental[df_incremental["K"] == k]
        df_k.drop(columns=["patient_num"]).to_excel(
            writer, sheet_name=f"K{k}", index=False
        )

print(f"Saved to {out_xlsx}")
