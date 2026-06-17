import pandas as pd
from scipy.stats import wilcoxon

# =========================
# 1. Load data
# =========================
in_file = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\统计分析\eval.xlsx"   # 你的原始长表
out_file = r"C:\Users\WS\Desktop\Esophagus\AAPM投稿\统计分析\Supplement_Pvalues.xlsx"

df = pd.read_excel(in_file)

# =========================
# 2. Configuration
# =========================
comparisons = [
    ("Uniform-Box", "Uniform-Mask"),
    ("Uniform-Mask", "Optimal-Mask"),
    ("Uniform-Box", "Optimal-Mask"),
]

comparison_names = [
    "Uniform-Box vs Uniform-Mask",
    "Uniform-Mask vs Optimal-Mask",
    "Uniform-Box vs Optimal-Mask",
]

metrics = [
    "Dice_All",
    "Dice_NoPrompt",
    "HD95_All",
    "HD95_NoPrompt",
]

Ks = sorted(df["K"].unique())

# =========================
# 3. Helper functions
# =========================
def paired_wilcoxon(sub_df, m1, m2, metric):
    d1 = sub_df[sub_df["Methods"] == m1].set_index("PatientID")[metric]
    d2 = sub_df[sub_df["Methods"] == m2].set_index("PatientID")[metric]
    common = d1.index.intersection(d2.index)
    if len(common) < 5:
        return None
    _, p = wilcoxon(d1.loc[common], d2.loc[common])
    return p

def format_p(p):
    if p is None:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# =========================
# 4. Compute table
# =========================
rows = []

for (m1, m2), cname in zip(comparisons, comparison_names):
    for K in Ks:
        sub = df[df["K"] == K]
        row = {
            "Comparison": cname,
            "K": K,
        }
        for metric in metrics:
            p = paired_wilcoxon(sub, m1, m2, metric)
            row[metric] = format_p(p)
        rows.append(row)

out_df = pd.DataFrame(rows)

# =========================
# 5. Sort rows
# =========================
out_df["Comparison"] = pd.Categorical(
    out_df["Comparison"],
    categories=comparison_names,
    ordered=True
)

out_df = out_df.sort_values(["Comparison", "K"])

# =========================
# 6. Reorder columns
# =========================
out_df = out_df[
    ["Comparison", "K", "Dice_All", "Dice_NoPrompt", "HD95_All", "HD95_NoPrompt"]
]

# =========================
# 7. Save
# =========================
out_df.to_excel(out_file, index=False)
