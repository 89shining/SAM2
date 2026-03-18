"""
统计每个患者中间提示层切片的指标：z位置，面积，面积变化率，中心位移
按K分sheet保存
"""

"""
合并 slice-level 指标 与 提示层信息，
并按 K=2-10 生成多 sheet Excel
"""

import ast
import pandas as pd


def build_prompt_slice_table_from_csv(prompt_csv):
    """
    从 oracle_patient_level.csv 中解析中间提示层
    返回 long-format DataFrame:
    patient_id | K | z
    """
    df = pd.read_csv(prompt_csv)

    rows = []

    for _, row in df.iterrows():
        patient_id = str(row["PatientID"]).strip()
        K = int(row["K"])  # 强制 int

        slices = ast.literal_eval(row["PromptSlices"])
        middle_slices = slices[2:]  # 去掉上下界

        for z in middle_slices:
            rows.append({
                "patient_id": patient_id,
                "K": K,
                "z": int(z),
            })

    return pd.DataFrame(rows)


def merge_prompt_with_slice_level_to_excel(
    slice_level_csv,
    prompt_csv,
    out_excel,
):
    """
    Sheet1 : All_Prompt
    Sheet2+: K2, K3, ..., Kn（按真实存在的 K）
    """

    # =========================
    # 1. 读取数据
    # =========================
    df_slice_all = pd.read_csv(slice_level_csv)
    df_prompt_long = build_prompt_slice_table_from_csv(prompt_csv)

    # ---------- 防 merge 出问题 ----------
    df_slice_all["patient_id"] = df_slice_all["patient_id"].astype(str).str.strip()
    df_slice_all["z"] = df_slice_all["z"].astype(int)

    df_prompt_long["patient_id"] = df_prompt_long["patient_id"].astype(str).str.strip()
    df_prompt_long["z"] = df_prompt_long["z"].astype(int)
    df_prompt_long["K"] = df_prompt_long["K"].astype(int)

    # =========================
    # 2. merge（只保留提示层）
    # =========================
    df_prompt_slice = df_prompt_long.merge(
        df_slice_all,
        on=["patient_id", "z"],
        how="inner"
    )

    if len(df_prompt_slice) == 0:
        raise RuntimeError("❌ 合并结果为空，请检查 patient_id / z 是否一致")

    # 再保险一次
    df_prompt_slice["K"] = df_prompt_slice["K"].astype(int)

    print("✅ Available K values:",
          sorted(df_prompt_slice["K"].unique()))

    # =========================
    # 3. 写 Excel（多 sheet）
    # =========================
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:

        # ---- Sheet 1：全部提示层 ----
        df_prompt_slice.to_excel(
            writer,
            sheet_name="All_Prompt",
            index=False
        )

        # ---- 后续：按真实存在的 K ----
        for K in sorted(df_prompt_slice["K"].unique()):
            df_k = df_prompt_slice[df_prompt_slice["K"] == K]
            sheet_name = f"K{K}"
            df_k.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    print(f"[OK] Excel saved to: {out_excel}")


# =========================
# 使用示例
# =========================
if __name__ == "__main__":

    slice_level_csv = r"C:\Users\dell\Desktop\Esophagus\AAPM投稿\Optimal_mask\slice_level_metrics.csv"
    prompt_csv = r"C:\Users\dell\Desktop\Esophagus\AAPM投稿\Optimal_mask\oracle_patient_level.csv"
    out_excel = r"C:\Users\dell\Desktop\Esophagus\AAPM投稿\Optimal_mask\Prompt_slice_level_metrics.xlsx"

    merge_prompt_with_slice_level_to_excel(
        slice_level_csv=slice_level_csv,
        prompt_csv=prompt_csv,
        out_excel=out_excel,
    )



