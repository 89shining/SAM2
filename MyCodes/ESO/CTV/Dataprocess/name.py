import pandas as pd
import re

def fix_patient_id_space_to_underscore(
    in_csv,
    out_csv,
    id_cols=("patient_id", "PatientID"),
):
    """
    修正 patient id 中的空格问题：
    p 55  -> p_55
    p   12 -> p_12

    - 只修指定的 id 列
    - 不影响其他字段
    """

    df = pd.read_csv(in_csv)

    def normalize_pid(x):
        if pd.isna(x):
            return x
        x = str(x).strip()
        # p 55 / p   55 -> p_55
        x = re.sub(r"\bp\s+(\d+)\b", r"p_\1", x)
        return x

    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_pid)

    df.to_csv(out_csv, index=False)
    print(f"[OK] fixed csv saved to: {out_csv}")


# =========================
# 使用示例
# =========================
if __name__ == "__main__":

    # 示例 1：修 slice_level_metrics.csv
    # fix_patient_id_space_to_underscore(
    #     in_csv=r"C:\Users\dell\Desktop\AAPM投稿\slice_level_metrics.csv",
    #     out_csv=r"C:\Users\dell\Desktop\AAPM投稿\slice_level_metrics_fixed.csv",
    #     id_cols=("patient_id",)
    # )

    # 示例 2：如果 oracle_patient_level.csv 里也有类似问题
    fix_patient_id_space_to_underscore(
        in_csv=r"C:\Users\dell\Desktop\AAPM投稿\oracle_patient_level.csv",
        out_csv=r"C:\Users\dell\Desktop\AAPM投稿\oracle_patient_level_fixed.csv",
        id_cols=("PatientID",)
    )
