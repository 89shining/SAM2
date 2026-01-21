import pandas as pd

# =========================
# 配置区
# =========================
IN_CSV  = r"C:\Users\dell\Desktop\oracle_patient_level55.csv"      # 你的原文件
OUT_CSV = r"C:\Users\dell\Desktop\oracle_patient_level83.csv"     # 排序后的文件

# =========================
# 读取
# =========================
df = pd.read_csv(IN_CSV)

# =========================
# 临时提取 PatientID 中的数值（仅用于排序）
# =========================
df["_pid_num"] = (
    df["PatientID"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(int)
)

# =========================
# 确保 K 用“数值”排序（不改原列）
# =========================
df["_K_num"] = df["K"].astype(int)

# =========================
# 排序：PatientID 数值 → K 数值
# =========================
df = df.sort_values(
    by=["_pid_num", "_K_num"],
    ascending=[True, True]
)

# =========================
# 删除临时列（保证其他都不动）
# =========================
df = df.drop(columns=["_pid_num", "_K_num"])

# =========================
# 保存
# =========================
df.to_csv(OUT_CSV, index=False)

print(f"✅ 排序完成，仅修正顺序：{OUT_CSV}")
