#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate esophageal CTV predictions for two settings:
1) boundary baseline (two prompts: lower/upper)
2) boundary + best third prompt

Per-patient metrics:
- Dice3D_boundary
- Dice3D_best3
- Delta_Dice3D
- Dice_noPrompt_boundary
- Dice_noPrompt_best3
- Delta_Dice_noPrompt
- HD95_boundary
- HD95_best3
- HD95_improve

Usage example:
python eval_boundary_vs_best3.py \
  --gt_dir "D:/path/to/gt" \
  --pred_boundary_dir "D:/path/to/boundary_pred" \
  --pred_best3_dir "D:/path/to/best3_pred" \
  --prompt2_excel "D:/path/to/mask_prompt_2/prompt_layer_search2.xlsx" \
  --prompt3_excel "D:/path/to/mask_prompt_3/prompt_layer_search3.xlsx" \
  --output_path "D:/path/to/eval_summary.csv"
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
# You can edit these defaults directly, or pass arguments from command line.
# Follow mask_prompt_2.py and mask_prompt_3.py default paths directly.
DEFAULT_DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260108/Data83")
DEFAULT_PRED_BOUNDARY_DIR = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_2/best_mask")
DEFAULT_PRED_BEST3_DIR = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_3/best_mask")
DEFAULT_PROMPT2_EXCEL = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_2/prompt_layer_search2.xlsx")
DEFAULT_PROMPT3_EXCEL = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_3/prompt_layer_search3.xlsx")
DEFAULT_OUTPUT_PATH = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/eval_boundary_vs_best3_summary.xlsx")


def normalize_id(value) -> str:
    """Normalize IDs from table/files to robust matching keys."""
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
    """Return filename stem compatible with .nii.gz and .nii."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def build_file_index(folder: Path) -> Dict[str, Path]:
    """Build case-insensitive map: normalized file stem -> absolute path."""
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
    """
    Build GT index using the same patient-folder style as mask_prompt_2/3:
    Data83/
      p_10/CTV.nii.gz
      p_11/CTV.nii.gz
      ...
    mapped to keys like CTV_010, CTV_011, ...
    """
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
    """Resolve patient file by Patient_ID."""
    key = normalize_id(patient_id).lower()
    if key not in file_index:
        raise FileNotFoundError(
            f"Patient '{patient_id}' not found in {folder_name}. "
            f"Expected file stem exactly matching Patient_ID."
        )
    return file_index[key]


def _find_sheet_with_columns(excel_path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    """Find the first sheet that contains all required columns."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if excel_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Only Excel is supported for multi-sheet prompt table: {excel_path}")

    sheets = pd.read_excel(excel_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        if all(col in df.columns for col in required_columns):
            print(f"[INFO] Using sheet '{sheet_name}' from {excel_path.name}")
            return df.copy()

    raise ValueError(
        f"Cannot find required columns {list(required_columns)} in any sheet of {excel_path}"
    )


def read_prompt_tables(prompt2_excel: Path, prompt3_excel: Path) -> pd.DataFrame:
    """
    Read prompt-layer stats exported by mask_prompt_2.py and mask_prompt_3.py.

    - prompt2_excel: provides Lower_Bound_ID / Upper_Bound_ID per patient
    - prompt3_excel: provides Best_Prompt_Slice_ID (best third prompt) per patient
    """
    df2 = _find_sheet_with_columns(
        prompt2_excel,
        ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID"],
    )
    df3 = _find_sheet_with_columns(
        prompt3_excel,
        ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID", "Best_Prompt_Slice_ID"],
    )

    df2 = df2[["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID"]].copy()
    df3 = df3[
        ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID", "Best_Prompt_Slice_ID"]
    ].copy()
    df3 = df3.rename(columns={"Best_Prompt_Slice_ID": "Best_Third_Prompt_ID"})

    for col in ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID"]:
        df2[col] = df2[col].apply(normalize_id)
    for col in ["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID", "Best_Third_Prompt_ID"]:
        df3[col] = df3[col].apply(normalize_id)

    # Merge by patient; boundary IDs prefer prompt2 table (fallback to prompt3 when missing).
    merged = pd.merge(
        df2,
        df3[["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID", "Best_Third_Prompt_ID"]],
        on="Patient_ID",
        how="outer",
        suffixes=("_p2", "_p3"),
    )

    merged["Lower_Bound_ID"] = merged["Lower_Bound_ID_p2"].where(
        merged["Lower_Bound_ID_p2"] != "", merged["Lower_Bound_ID_p3"]
    )
    merged["Upper_Bound_ID"] = merged["Upper_Bound_ID_p2"].where(
        merged["Upper_Bound_ID_p2"] != "", merged["Upper_Bound_ID_p3"]
    )

    # Consistency check when both prompt2 and prompt3 carry non-empty boundary ids.
    both_lower = (merged["Lower_Bound_ID_p2"] != "") & (merged["Lower_Bound_ID_p3"] != "")
    both_upper = (merged["Upper_Bound_ID_p2"] != "") & (merged["Upper_Bound_ID_p3"] != "")
    bad_lower = merged[both_lower & (merged["Lower_Bound_ID_p2"] != merged["Lower_Bound_ID_p3"])]
    bad_upper = merged[both_upper & (merged["Upper_Bound_ID_p2"] != merged["Upper_Bound_ID_p3"])]
    if len(bad_lower) > 0 or len(bad_upper) > 0:
        raise ValueError(
            "Boundary IDs are inconsistent between prompt2 and prompt3 excel files. "
            "Please check Lower_Bound_ID / Upper_Bound_ID."
        )

    out = merged[["Patient_ID", "Lower_Bound_ID", "Upper_Bound_ID", "Best_Third_Prompt_ID"]].copy()
    out = out.dropna(how="any")
    out = out[(out["Patient_ID"] != "") & (out["Best_Third_Prompt_ID"] != "")]
    return out


def read_nii(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    """Read NIfTI with SimpleITK and return array [Z, H, W] + image object."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    img = sitk.ReadImage(str(path))
    arr_zyx = sitk.GetArrayFromImage(img)  # [Z, H, W]
    return arr_zyx, img


def to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def dice_3d(pred: np.ndarray, gt: np.ndarray) -> float:
    """3D Dice with empty-mask handling."""
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
    """Remove given z-slices from mask for Dice_noPrompt computation."""
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
    """Compute Dice on volume after removing prompt z-slices from both pred and gt."""
    pred_wo = remove_prompt_slices(pred_zyx, prompt_ids, patient_id)
    gt_wo = remove_prompt_slices(gt_zyx, prompt_ids, patient_id)
    return dice_3d(pred_wo, gt_wo)


def hd95_3d(pred_zyx: np.ndarray, gt_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    """Compute 3D HD95 (NaN if pred or gt is empty)."""
    pred_bool = pred_zyx.astype(bool)
    gt_bool = gt_zyx.astype(bool)

    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return float("nan")

    # medpy expects spacing in array axis order -> [z, y, x]
    return float(hd95(pred_bool, gt_bool, voxelspacing=spacing_zyx))


def ensure_same_shape(patient_id: str, gt: np.ndarray, pred_boundary: np.ndarray, pred_best3: np.ndarray) -> None:
    """Raise error if any prediction shape mismatches GT shape."""
    if pred_boundary.shape != gt.shape or pred_best3.shape != gt.shape:
        raise ValueError(
            f"[{patient_id}] Shape mismatch detected:\n"
            f"  GT shape:       {gt.shape}\n"
            f"  boundary shape: {pred_boundary.shape}\n"
            f"  best3 shape:    {pred_best3.shape}"
        )


def parse_prompt_ids(row: pd.Series) -> Tuple[int, int, int]:
    """Parse prompt layer IDs to int."""
    patient_id = row["Patient_ID"]
    try:
        lower = int(row["Lower_Bound_ID"])
        upper = int(row["Upper_Bound_ID"])
        third = int(row["Best_Third_Prompt_ID"])
    except ValueError as exc:
        raise ValueError(f"[{patient_id}] Prompt IDs must be integer-like values: {row.to_dict()}") from exc
    return lower, upper, third


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save to CSV or Excel based on extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output format: {output_path}. "
            f"Use .csv or .xlsx"
        )


def main(
    data_root: Path,
    pred_boundary_dir: Path,
    pred_best3_dir: Path,
    prompt2_excel: Path,
    prompt3_excel: Path,
    output_path: Path,
) -> None:
    prompt_df = read_prompt_tables(prompt2_excel, prompt3_excel)

    gt_index = build_gt_index_from_data_root(data_root)
    boundary_index = build_file_index(pred_boundary_dir)
    best3_index = build_file_index(pred_best3_dir)

    rows: List[dict] = []

    for _, row in prompt_df.iterrows():
        patient_id = row["Patient_ID"]
        lower_id, upper_id, third_id = parse_prompt_ids(row)

        gt_path = resolve_patient_file(patient_id, gt_index, "GT folder")
        boundary_path = resolve_patient_file(patient_id, boundary_index, "boundary prediction folder")
        best3_path = resolve_patient_file(patient_id, best3_index, "best3 prediction folder")

        gt_zyx, gt_img = read_nii(gt_path)
        pred_boundary_zyx, _ = read_nii(boundary_path)
        pred_best3_zyx, _ = read_nii(best3_path)

        gt_zyx = to_binary(gt_zyx)
        pred_boundary_zyx = to_binary(pred_boundary_zyx)
        pred_best3_zyx = to_binary(pred_best3_zyx)

        ensure_same_shape(patient_id, gt_zyx, pred_boundary_zyx, pred_best3_zyx)

        spacing_xyz = gt_img.GetSpacing()  # (x, y, z)
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

        dice3d_boundary = dice_3d(pred_boundary_zyx, gt_zyx)
        dice3d_best3 = dice_3d(pred_best3_zyx, gt_zyx)
        delta_dice3d = dice3d_best3 - dice3d_boundary

        prompt_boundary = [lower_id, upper_id]
        prompt_best3 = [lower_id, upper_id, third_id]

        dice_np_boundary = dice_no_prompt(
            pred_boundary_zyx, gt_zyx, prompt_boundary, patient_id
        )
        dice_np_best3 = dice_no_prompt(
            pred_best3_zyx, gt_zyx, prompt_best3, patient_id
        )
        delta_dice_np = dice_np_best3 - dice_np_boundary

        hd95_boundary = hd95_3d(pred_boundary_zyx, gt_zyx, spacing_zyx)
        hd95_best3 = hd95_3d(pred_best3_zyx, gt_zyx, spacing_zyx)
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
    print(f"[INFO] Done. Saved {len(result_df)} patients to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate boundary baseline vs boundary+best3 CTV predictions"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root like Data83 containing p_xxx/CTV.nii.gz (same as mask_prompt_2/3).",
    )
    parser.add_argument("--pred_boundary_dir", type=Path, default=DEFAULT_PRED_BOUNDARY_DIR)
    parser.add_argument("--pred_best3_dir", type=Path, default=DEFAULT_PRED_BEST3_DIR)
    parser.add_argument("--prompt2_excel", type=Path, default=DEFAULT_PROMPT2_EXCEL)
    parser.add_argument("--prompt3_excel", type=Path, default=DEFAULT_PROMPT3_EXCEL)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_root=args.data_root,
        pred_boundary_dir=args.pred_boundary_dir,
        pred_best3_dir=args.pred_best3_dir,
        prompt2_excel=args.prompt2_excel,
        prompt3_excel=args.prompt3_excel,
        output_path=args.output_path,
    )
