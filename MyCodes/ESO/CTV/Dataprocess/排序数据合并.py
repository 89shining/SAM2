import pandas as pd

# =========================
# 配置区
# =========================
IN_CSV  = r"C:\Users\dell\Desktop\oracle_patient_level.csv"
OUT_CSV = r"C:\Users\dell\Desktop\oracle_patient_level_id10_sorted.csv"

# =========================
# 读取
# =========================
df = pd.read_csv(IN_CSV)

# =========================
# 1. 处理 PatientID
#    p 0 / p 10 → 数值 → +55 → p 55 / p 65
# =========================
df["pid_num"] = (
    df["PatientID"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(int)
)

df["pid_num_new"] = df["pid_num"] + 55
df["PatientID"] = "p " + df["pid_num_new"].astype(str)

# =========================
# 2. 确保 K 是整数（非常关键）
# =========================
df["K"] = df["K"].astype(int)

# =========================
# 3. 按 PatientID(数值) + K(数值) 排序
# =========================
df = df.sort_values(
    by=["pid_num_new", "K"],
    ascending=[True, True]
)

# =========================
# 4. 清理中间列
# =========================
df = df.drop(columns=["pid_num", "pid_num_new"])

# =========================
# 5. 保存
# =========================
df.to_csv(OUT_CSV, index=False)

print(f"✅ 已保存：{OUT_CSV}")
