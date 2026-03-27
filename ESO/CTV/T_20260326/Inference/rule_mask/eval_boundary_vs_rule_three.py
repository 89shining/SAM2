#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate rule-based third-prompt results against boundary baseline.

This script follows the same metric/table format as:
  Inference/oracle_mask/eval_boundary_vs_best3.py

It generates 3 separate result files (same columns) for:
- middle-rule third prompt
- max-area-rule third prompt
- max-change-rule third prompt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

try:
    from medpy.metric.binary import hd95
except ImportError as exc:
    raise ImportError(
        "medpy is required for HD95. Please install it first: pip install medpy"
    ) from exc


# ======================================================
# ================= Configurable Paths =================
# ======================================================
# Paths aligned with mask_prompt_2.py and mask_prompt_rule.py defaults.
DEFAULT_DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260108/Data83")
DEFAULT_PRED_BOUNDARY_DIR = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_2/best_mask")

DEFAULT_RULE_MIDDLE_DIR = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/rule_middle_mask")
DEFAULT_RULE_MAXAREA_DIR = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/rule_maxarea_mask")
# NOTE: this name follows your current mask_prompt_rule.py output folder.
DEFAULT_RULE_MAXCHANGE_DIR = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/rule_maxarea_change_mask")

DEFAULT_PROMPT2_EXCEL = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_2/prompt_layer_search2.xlsx")
DEFAULT_RULE_SUMMARY_XLSX = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/prompt_rule_summary.xlsx")

DEFAULT_OUT_MIDDLE = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/eval_boundary_vs_rule_middle.xlsx")
DEFAULT_OUT_MAXAREA = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/eval_boundary_vs_rule_maxarea.xlsx")
DEFAULT_OUT_MAXCHANGE = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/rule_mask/eval_boundary_vs_rule_maxchange.xlsx")


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return str(int(value)).strip()

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value)).strip()
        return str(value).strip()

    text = str(value).strip()
    if text.endswith(".0"):
        numeric_part = text[:-2]
        if numeric_part.isdigit():
            return numeric_part
    return text


def nii_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def build_file_index(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in: {folder}")

    index: Dict[str, Path] = {}
    for f in files:
        key = normalize_id(nii_stem(f)).lower()
        if key in index:
            raise ValueError(
                f"Duplicate patient key in folder '{folder}': '{key}'\n"
                f"  - {index[key]}\n"
                f"  - {f}"
            )
        index[key] = f
    return index


def build_gt_index_from_data_root(data_root: Path, gt_name: str = "CTV.nii.gz") -> Dict[str, Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    patient_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if len(patient_dirs) == 0:
        raise FileNotFoundError(f"No patient directories found in: {data_root}")

    index: Dict[str, Path] = {}
    for pdir in patient_dirs:
        m = re.fullmatch(r"p_(\d+)", pdir.name)
        if m is None:
            continue

        pid_num = int(m.group(1))
        patient_id = f"CTV_{pid_num:03d}"
        gt_path = pdir / gt_name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT file for {patient_id}: {gt_path}")

        key = normalize_id(patient_id).lower()
        if key in index:
            raise ValueError(f"Duplicate GT patient key detected: {patient_id}")
        index[key] = gt_path

    if len(index) == 0:
        raise ValueError(
            f"No valid patient folders like 'p_10' with '{gt_name}' found in: {data_root}"
        )
    return index


def resolve_patient_file(patient_id: str, file_index: Dict[str, Path], folder_name: str) -> Path:
    key = normalize_id(patient_id).lower()
    if key not in file_index:
        raise FileNotFoundError(
            f"Patient '{patient_id}' not found in {folder_name}. "
            f"Expected file stem exactly matching Patient_ID."
        )
    return file_index[key]


def _find_sheet_with_columns(excel_path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if excel_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Only Excel is supported: {excel_path}")

    sheets = pd.read_excel(excel_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        if all(col in df.columns for col in required_columns):
            print(f"[INFO] Using sheet '{sheet_name}' from {excel_path.name}")
            return df.copy()

    raise ValueError(
        f"Cannot find required columns {list(required_columns)} in any sheet of {excel_path}"
    )


def read_prompt2_bounds(prompt2_excel: Path) -> pd.DataFrame:
    df2 = _find_sheet_with_columns(
        prompt2_excel,
        ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID"],
    )
    df2 = df2[["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID"]].copy()

    for col in ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID"]:
        df2[col] = df2[col].apply(normalize_id)

    df2 = df2.dropna(how="any")
    df2 = df2[(df2["Patient_ID"] != "") & (df2["Lower_Bound_ID"] != "") & (df2["Upper_Bound_ID"] != "")]
    df2 = df2.drop_duplicates(subset=["Patient_ID"], keep="first")
    return df2


def read_rule_summary(rule_summary_xlsx: Path) -> pd.DataFrame:
    df_rule = _find_sheet_with_columns(
        rule_summary_xlsx,
        [
            "Patient_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Rule_Middle_Slice_ID",
            "Rule_MaxArea_Slice_ID",
            "Rule_MaxChange_Slice_ID",
        ],
    )

    keep_cols = [
        "Patient_ID",
        "Lower_Bound_ID",
        "Upper_Bound_ID",
        "Rule_Middle_Slice_ID",
        "Rule_MaxArea_Slice_ID",
        "Rule_MaxChange_Slice_ID",
    ]
    df_rule = df_rule[keep_cols].copy()
    for c in keep_cols:
        df_rule[c] = df_rule[c].apply(normalize_id)

    df_rule = df_rule.dropna(how="any")
    df_rule = df_rule[df_rule["Patient_ID"] != ""]
    df_rule = df_rule.drop_duplicates(subset=["Patient_ID"], keep="first")
    return df_rule


def merge_prompts(prompt2_excel: Path, rule_summary_xlsx: Path) -> pd.DataFrame:
    df2 = read_prompt2_bounds(prompt2_excel)
    dfr = read_rule_summary(rule_summary_xlsx)

    merged = pd.merge(
        df2,
        dfr,
        on="Patient_ID",
        how="inner",
        suffixes=("_p2", "_rule"),
    )

    # Boundary consistency check (if both present and non-empty).
    for bound in ["Lower_Bound_ID", "Upper_Bound_ID"]:
        a = f"{bound}_p2"
        b = f"{bound}_rule"
        mask = (merged[a] != "") & (merged[b] != "") & (merged[a] != merged[b])
        if mask.any():
            bad_ids = merged.loc[mask, "Patient_ID"].tolist()
            raise ValueError(f"{bound} mismatch between prompt2 and rule summary for patients: {bad_ids[:10]}")

    merged["Lower_Bound_ID"] = merged["Lower_Bound_ID_p2"].where(
        merged["Lower_Bound_ID_p2"] != "", merged["Lower_Bound_ID_rule"]
    )
    merged["Upper_Bound_ID"] = merged["Upper_Bound_ID_p2"].where(
        merged["Upper_Bound_ID_p2"] != "", merged["Upper_Bound_ID_rule"]
    )

    out = merged[
        [
            "Patient_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Rule_Middle_Slice_ID",
            "Rule_MaxArea_Slice_ID",
            "Rule_MaxChange_Slice_ID",
        ]
    ].copy()

    return out


def read_nii(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    img = sitk.ReadImage(str(path))
    arr_zyx = sitk.GetArrayFromImage(img)
    return arr_zyx, img


def to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    pred_sum = int(pred_bool.sum())
    gt_sum = int(gt_bool.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    return float(2.0 * intersection / (pred_sum + gt_sum))


def remove_prompt_slices(mask_zyx: np.ndarray, prompt_ids: Sequence[int], patient_id: str) -> np.ndarray:
    if mask_zyx.ndim != 3:
        raise ValueError(f"[{patient_id}] mask must be 3D [Z,H,W], got shape {mask_zyx.shape}")

    z_dim = mask_zyx.shape[0]
    unique_ids = sorted(set(int(x) for x in prompt_ids))

    for z in unique_ids:
        if z < 0 or z >= z_dim:
            raise IndexError(
                f"[{patient_id}] Prompt slice index out of range: z={z}, valid=[0, {z_dim - 1}]"
            )

    keep = np.ones(z_dim, dtype=bool)
    keep[unique_ids] = False
    return mask_zyx[keep]


def dice_no_prompt(pred_zyx: np.ndarray, gt_zyx: np.ndarray, prompt_ids: Sequence[int], patient_id: str) -> float:
    pred_wo = remove_prompt_slices(pred_zyx, prompt_ids, patient_id)
    gt_wo = remove_prompt_slices(gt_zyx, prompt_ids, patient_id)
    return dice_3d(pred_wo, gt_wo)


def hd95_3d(pred_zyx: np.ndarray, gt_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    pred_bool = pred_zyx.astype(bool)
    gt_bool = gt_zyx.astype(bool)

    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return float("nan")

    return float(hd95(pred_bool, gt_bool, voxelspacing=spacing_zyx))


def ensure_same_shape(patient_id: str, gt: np.ndarray, pred_boundary: np.ndarray, pred_rule: np.ndarray) -> None:
    if pred_boundary.shape != gt.shape or pred_rule.shape != gt.shape:
        raise ValueError(
            f"[{patient_id}] Shape mismatch detected:\n"
            f"  GT shape:       {gt.shape}\n"
            f"  boundary shape: {pred_boundary.shape}\n"
            f"  rule shape:     {pred_rule.shape}"
        )


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}. Use .csv or .xlsx")


def evaluate_one_rule(
    merged_prompt_df: pd.DataFrame,
    rule_col_name: str,
    pred_rule_dir: Path,
    output_path: Path,
    gt_index: Dict[str, Path],
    boundary_index: Dict[str, Path],
) -> None:
    rule_index = build_file_index(pred_rule_dir)
    rows: List[dict] = []

    for _, row in merged_prompt_df.iterrows():
        patient_id = row["Patient_ID"]
        try:
            lower_id = int(row["Lower_Bound_ID"])
            upper_id = int(row["Upper_Bound_ID"])
            third_id = int(row[rule_col_name])
        except ValueError as exc:
            raise ValueError(f"[{patient_id}] Prompt IDs must be integer-like values") from exc

        gt_path = resolve_patient_file(patient_id, gt_index, "GT folder")
        boundary_path = resolve_patient_file(patient_id, boundary_index, "boundary prediction folder")
        rule_path = resolve_patient_file(patient_id, rule_index, f"{rule_col_name} prediction folder")

        gt_zyx, gt_img = read_nii(gt_path)
        pred_boundary_zyx, _ = read_nii(boundary_path)
        pred_rule_zyx, _ = read_nii(rule_path)

        gt_zyx = to_binary(gt_zyx)
        pred_boundary_zyx = to_binary(pred_boundary_zyx)
        pred_rule_zyx = to_binary(pred_rule_zyx)

        ensure_same_shape(patient_id, gt_zyx, pred_boundary_zyx, pred_rule_zyx)

        spacing_xyz = gt_img.GetSpacing()
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

        dice3d_boundary = dice_3d(pred_boundary_zyx, gt_zyx)
        dice3d_best3 = dice_3d(pred_rule_zyx, gt_zyx)
        delta_dice3d = dice3d_best3 - dice3d_boundary

        prompt_boundary = [lower_id, upper_id]
        prompt_best3 = [lower_id, upper_id, third_id]

        dice_np_boundary = dice_no_prompt(pred_boundary_zyx, gt_zyx, prompt_boundary, patient_id)
        dice_np_best3 = dice_no_prompt(pred_rule_zyx, gt_zyx, prompt_best3, patient_id)
        delta_dice_np = dice_np_best3 - dice_np_boundary

        hd95_boundary = hd95_3d(pred_boundary_zyx, gt_zyx, spacing_zyx)
        hd95_best3 = hd95_3d(pred_rule_zyx, gt_zyx, spacing_zyx)
        hd95_improve = hd95_boundary - hd95_best3

        rows.append(
            {
                "Patient_ID": patient_id,
                "Lower_Bound_ID": lower_id,
                "Upper_Bound_ID": upper_id,
                "Best_Third_Prompt_ID": third_id,
                "Dice3D_boundary": dice3d_boundary,
                "Dice3D_best3": dice3d_best3,
                "Delta_Dice3D": delta_dice3d,
                "Dice_noPrompt_boundary": dice_np_boundary,
                "Dice_noPrompt_best3": dice_np_best3,
                "Delta_Dice_noPrompt": delta_dice_np,
                "HD95_boundary": hd95_boundary,
                "HD95_best3": hd95_best3,
                "HD95_improve": hd95_improve,
            }
        )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(by="Patient_ID", ascending=True).reset_index(drop=True)
    save_results(result_df, output_path)
    print(f"[INFO] Saved {len(result_df)} rows -> {output_path}")


def main(
    data_root: Path,
    pred_boundary_dir: Path,
    rule_middle_dir: Path,
    rule_maxarea_dir: Path,
    rule_maxchange_dir: Path,
    prompt2_excel: Path,
    rule_summary_xlsx: Path,
    out_middle: Path,
    out_maxarea: Path,
    out_maxchange: Path,
) -> None:
    merged_prompt_df = merge_prompts(prompt2_excel, rule_summary_xlsx)

    gt_index = build_gt_index_from_data_root(data_root)
    boundary_index = build_file_index(pred_boundary_dir)

    evaluate_one_rule(
        merged_prompt_df=merged_prompt_df,
        rule_col_name="Rule_Middle_Slice_ID",
        pred_rule_dir=rule_middle_dir,
        output_path=out_middle,
        gt_index=gt_index,
        boundary_index=boundary_index,
    )

    evaluate_one_rule(
        merged_prompt_df=merged_prompt_df,
        rule_col_name="Rule_MaxArea_Slice_ID",
        pred_rule_dir=rule_maxarea_dir,
        output_path=out_maxarea,
        gt_index=gt_index,
        boundary_index=boundary_index,
    )

    evaluate_one_rule(
        merged_prompt_df=merged_prompt_df,
        rule_col_name="Rule_MaxChange_Slice_ID",
        pred_rule_dir=rule_maxchange_dir,
        output_path=out_maxchange,
        gt_index=gt_index,
        boundary_index=boundary_index,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate three rule-based third-prompt results vs boundary baseline"
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred_boundary_dir", type=Path, default=DEFAULT_PRED_BOUNDARY_DIR)

    parser.add_argument("--rule_middle_dir", type=Path, default=DEFAULT_RULE_MIDDLE_DIR)
    parser.add_argument("--rule_maxarea_dir", type=Path, default=DEFAULT_RULE_MAXAREA_DIR)
    parser.add_argument("--rule_maxchange_dir", type=Path, default=DEFAULT_RULE_MAXCHANGE_DIR)

    parser.add_argument("--prompt2_excel", type=Path, default=DEFAULT_PROMPT2_EXCEL)
    parser.add_argument("--rule_summary_xlsx", type=Path, default=DEFAULT_RULE_SUMMARY_XLSX)

    parser.add_argument("--out_middle", type=Path, default=DEFAULT_OUT_MIDDLE)
    parser.add_argument("--out_maxarea", type=Path, default=DEFAULT_OUT_MAXAREA)
    parser.add_argument("--out_maxchange", type=Path, default=DEFAULT_OUT_MAXCHANGE)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_root=args.data_root,
        pred_boundary_dir=args.pred_boundary_dir,
        rule_middle_dir=args.rule_middle_dir,
        rule_maxarea_dir=args.rule_maxarea_dir,
        rule_maxchange_dir=args.rule_maxchange_dir,
        prompt2_excel=args.prompt2_excel,
        rule_summary_xlsx=args.rule_summary_xlsx,
        out_middle=args.out_middle,
        out_maxarea=args.out_maxarea,
        out_maxchange=args.out_maxchange,
    )
