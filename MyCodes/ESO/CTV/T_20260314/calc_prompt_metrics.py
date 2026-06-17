import ast
import csv
import os
import re

import numpy as np
import SimpleITK as sitk
from openpyxl import Workbook
from scipy.ndimage import binary_erosion, distance_transform_edt

# =========================
# Modify paths here
# =========================
PRED_DIR = "/home/wusi/SAM2/SAM2data/Eso/20260315/TestResult"
GT_DIR = "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/labelsTs"
PROMPT_CSV = "/home/wusi/SAM2/SAM2data/Eso/20260315/TestResult/prompt_layers_info.csv"
OUT_XLSX = "/home/wusi/SAM2/SAM2data/Eso/20260315/TestResult/SAM2_metrics_summary.xlsx"


def load_binary_mask(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    arr = (arr > 0).astype(np.uint8)

    # SimpleITK spacing: (x, y, z), scipy EDT sampling uses axis order (z, y, x)
    sx, sy, sz = img.GetSpacing()
    sampling_zyx = (float(sz), float(sy), float(sx))
    return arr, sampling_zyx


def dice_3d(pred, gt, eps=1e-8):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def surface_distances(pred, gt, sampling):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float64)
    if not pred.any() or not gt.any():
        return np.array([np.inf], dtype=np.float64)

    footprint = np.ones((3, 3, 3), dtype=bool)
    pred_eroded = binary_erosion(pred, structure=footprint, border_value=0)
    gt_eroded = binary_erosion(gt, structure=footprint, border_value=0)

    pred_surface = np.logical_xor(pred, pred_eroded)
    gt_surface = np.logical_xor(gt, gt_eroded)

    if not pred_surface.any():
        pred_surface = pred
    if not gt_surface.any():
        gt_surface = gt

    dist_to_gt = distance_transform_edt(~gt_surface, sampling=sampling)
    dist_to_pred = distance_transform_edt(~pred_surface, sampling=sampling)

    d_pred_gt = dist_to_gt[pred_surface]
    d_gt_pred = dist_to_pred[gt_surface]
    return np.concatenate([d_pred_gt, d_gt_pred]).astype(np.float64)


def hd95_3d(pred, gt, sampling):
    d = surface_distances(pred, gt, sampling)
    if not np.isfinite(d).all():
        return np.nan
    return float(np.percentile(d, 95))


def parse_prompt_layers(prompt_csv):
    prompt_map = {}
    with open(prompt_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_raw = str(row.get("patient", "")).strip()
            if not patient_raw:
                continue

            m = re.search(r"(\d+)$", patient_raw)
            if m is None:
                continue
            idx = int(m.group(1))
            pid = f"CTV_{idx:03d}"

            layers_raw = row.get("prompt_layers_abs", "")
            try:
                layers = ast.literal_eval(layers_raw) if layers_raw else []
            except Exception:
                layers = []

            layers = sorted(set(int(x) for x in layers))
            prompt_map[pid] = layers
    return prompt_map


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


def main():
    prompt_map = parse_prompt_layers(PROMPT_CSV)
    pred_files = sorted(
        [x for x in os.listdir(PRED_DIR) if x.lower().endswith(".nii") or x.lower().endswith(".nii.gz")]
    )

    rows = []
    for fn in pred_files:
        pid = re.sub(r"\.nii(\.gz)?$", "", fn, flags=re.IGNORECASE)
        pred_path = os.path.join(PRED_DIR, fn)
        gt_path = os.path.join(GT_DIR, f"{pid}.nii.gz")
        if not os.path.exists(gt_path):
            continue

        pred, sampling = load_binary_mask(pred_path)
        gt, _ = load_binary_mask(gt_path)
        if pred.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch for {pid}: pred{pred.shape} vs gt{gt.shape}")

        prompt_layers = [z for z in prompt_map.get(pid, []) if 0 <= z < pred.shape[0]]

        dice_all = dice_3d(pred, gt)
        hd95_all = hd95_3d(pred, gt, sampling)

        pred_no_prompt = pred.copy()
        gt_no_prompt = gt.copy()
        if prompt_layers:
            pred_no_prompt[prompt_layers, :, :] = 0
            gt_no_prompt[prompt_layers, :, :] = 0

        dice_no_prompt = dice_3d(pred_no_prompt, gt_no_prompt)
        hd95_no_prompt = hd95_3d(pred_no_prompt, gt_no_prompt, sampling)

        rows.append(
            {
                "Patient_ID": pid,
                "Prompt_Num": len(prompt_layers),
                "Dice_All": dice_all,
                "HD95_All": hd95_all,
                "Dice_NoPrompt": dice_no_prompt,
                "HD95_NoPrompt": hd95_no_prompt,
            }
        )

    rows.sort(
        key=lambda r: int(re.search(r"(\d+)$", r["Patient_ID"]).group(1))
        if re.search(r"(\d+)$", r["Patient_ID"])
        else r["Patient_ID"]
    )

    prompt_mean, _ = mean_std([r["Prompt_Num"] for r in rows])
    dice_all_mean, dice_all_std = mean_std([r["Dice_All"] for r in rows])
    hd95_all_mean, hd95_all_std = mean_std([r["HD95_All"] for r in rows])
    dice_np_mean, dice_np_std = mean_std([r["Dice_NoPrompt"] for r in rows])
    hd95_np_mean, hd95_np_std = mean_std([r["HD95_NoPrompt"] for r in rows])

    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    ws.append(["Patient_ID", "Prompt_Num", "Dice_All", "HD95_All", "Dice_NoPrompt", "HD95_NoPrompt"])

    for r in rows:
        ws.append(
            [
                r["Patient_ID"],
                r["Prompt_Num"],
                f"{r['Dice_All']:.2f}",
                f"{r['HD95_All']:.2f}" if np.isfinite(r["HD95_All"]) else "nan",
                f"{r['Dice_NoPrompt']:.2f}",
                f"{r['HD95_NoPrompt']:.2f}" if np.isfinite(r["HD95_NoPrompt"]) else "nan",
            ]
        )

    ws.append(
        [
            "Mean±Std",
            f"{prompt_mean:.2f}",
            f"{dice_all_mean:.2f}±{dice_all_std:.2f}",
            f"{hd95_all_mean:.2f}±{hd95_all_std:.2f}",
            f"{dice_np_mean:.2f}±{dice_np_std:.2f}",
            f"{hd95_np_mean:.2f}±{hd95_np_std:.2f}",
        ]
    )

    wb.save(OUT_XLSX)
    print(f"Saved XLSX: {OUT_XLSX}")


if __name__ == "__main__":
    main()
