#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
import zipfile
from xml.sax.saxutils import escape
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont


def parse_patient_id(filename: str) -> str | None:
    m = re.fullmatch(r"(CTV_\d+)\.nii(\.gz)?", filename)
    if not m:
        return None
    return m.group(1)


def read_nii(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    return arr, img


def to_binary(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.uint8)


def dice_2d(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def hd95_2d(pred: np.ndarray, gt: np.ndarray, spacing_xy: Tuple[float, float]) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)

    p_sum = int(p.sum())
    g_sum = int(g.sum())

    if p_sum == 0 and g_sum == 0:
        return 0.0
    if p_sum == 0 or g_sum == 0:
        return float("nan")

    p_img = sitk.GetImageFromArray(p.astype(np.uint8))
    g_img = sitk.GetImageFromArray(g.astype(np.uint8))
    p_img.SetSpacing((float(spacing_xy[0]), float(spacing_xy[1])))
    g_img.SetSpacing((float(spacing_xy[0]), float(spacing_xy[1])))

    p_surface = sitk.LabelContour(p_img)
    g_surface = sitk.LabelContour(g_img)

    dmap_to_g = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            g_img,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )
    dmap_to_p = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            p_img,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    p_surface_arr = sitk.GetArrayViewFromImage(p_surface) > 0
    g_surface_arr = sitk.GetArrayViewFromImage(g_surface) > 0
    dmap_to_g_arr = sitk.GetArrayViewFromImage(dmap_to_g)
    dmap_to_p_arr = sitk.GetArrayViewFromImage(dmap_to_p)

    dist_p_to_g = dmap_to_g_arr[p_surface_arr]
    dist_g_to_p = dmap_to_p_arr[g_surface_arr]

    if dist_p_to_g.size == 0 and dist_g_to_p.size == 0:
        return float("nan")

    all_dist = np.concatenate([dist_p_to_g, dist_g_to_p]).astype(np.float64, copy=False)
    return float(np.percentile(all_dist, 95))


def find_gt_bounds(gt_zyx: np.ndarray) -> Tuple[int, int]:
    # Return (lower_id, upper_id), where lower_id < upper_id.
    pos = np.where(gt_zyx.reshape(gt_zyx.shape[0], -1).any(axis=1))[0]
    if pos.size == 0:
        return -1, -1
    return int(pos.min()), int(pos.max())


def collect_cases(gt_dir: Path, pred_dirs: Dict[str, Path]) -> Dict[str, Dict[str, Path]]:
    gt_map: Dict[str, Path] = {}
    for f in sorted(gt_dir.glob("*.nii*")):
        pid = parse_patient_id(f.name)
        if pid is not None:
            gt_map[pid] = f

    pred_maps: Dict[str, Dict[str, Path]] = {}
    for name, d in pred_dirs.items():
        m: Dict[str, Path] = {}
        for f in sorted(d.glob("*.nii*")):
            pid = parse_patient_id(f.name)
            if pid is not None:
                m[pid] = f
        pred_maps[name] = m

    shared = set(gt_map.keys())
    for m in pred_maps.values():
        shared &= set(m.keys())

    cases: Dict[str, Dict[str, Path]] = {}
    for pid in sorted(shared, key=lambda x: int(x.split("_")[-1])):
        item = {"gt": gt_map[pid]}
        for name, m in pred_maps.items():
            item[name] = m[pid]
        cases[pid] = item

    return cases


def build_xticks(num_layers: int, tick_step: int) -> List[int]:
    ticks = list(range(tick_step, num_layers + 1, tick_step))
    if not ticks or ticks[-1] != num_layers:
        ticks.append(num_layers)
    return ticks


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def plot_per_patient(
    patient_id: str,
    metric_name: str,
    values_by_model: Dict[str, List[float]],
    out_path: Path,
    tick_step: int,
) -> None:
    _plot_with_pillow(patient_id, metric_name, values_by_model, out_path, tick_step)


def _plot_with_pillow(
    patient_id: str,
    metric_name: str,
    values_by_model: Dict[str, List[float]],
    out_path: Path,
    tick_step: int,
) -> None:
    num_layers = len(next(iter(values_by_model.values())))
    xticks = build_xticks(num_layers=num_layers, tick_step=tick_step)

    width, height = 1550, 980
    ml, mt, mr, mb = 130, 95, 60, 250
    pw = width - ml - mr
    ph = height - mt - mb

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_tick = load_font(28)
    font_axis = load_font(34)
    font_title = load_font(40)
    font_legend = load_font(28)

    def finite_values(vs: List[float]) -> List[float]:
        arr = np.asarray(vs, dtype=np.float64)
        return arr[np.isfinite(arr)].tolist()

    all_finite = []
    for vals in values_by_model.values():
        all_finite.extend(finite_values(vals))

    if metric_name.lower().startswith("2d dice"):
        y_min, y_max = 0.0, 1.0
    elif all_finite:
        y_min = min(all_finite)
        y_max = max(all_finite)
        if math.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = 0.0, 1.0

    def x_to_px(xi: int) -> float:
        if num_layers <= 1:
            return float(ml + pw / 2)
        return float(ml + (xi - 1) / (num_layers - 1) * pw)

    def y_to_px(yv: float) -> float:
        if not np.isfinite(yv):
            return float("nan")
        return float(mt + (y_max - yv) / (y_max - y_min) * ph)

    # Axes
    draw.line((ml, mt, ml, mt + ph), fill=(30, 30, 30), width=2)
    draw.line((ml, mt + ph, ml + pw, mt + ph), fill=(30, 30, 30), width=2)

    # Y grid/ticks
    y_ticks = np.linspace(y_min, y_max, 6)
    for yv in y_ticks:
        yp = y_to_px(float(yv))
        draw.line((ml, yp, ml + pw, yp), fill=(235, 235, 235), width=1)
        txt = f"{yv:.2f}"
        tw, th = draw.textbbox((0, 0), txt, font=font_tick)[2:]
        draw.text((ml - tw - 14, yp - th / 2), txt, fill=(40, 40, 40), font=font_tick)

    # X ticks: keep fixed segments, always include last layer.
    for xt in xticks:
        xp = x_to_px(xt)
        draw.line((xp, mt + ph, xp, mt + ph + 12), fill=(30, 30, 30), width=2)
        t = str(xt)
        tw, th = draw.textbbox((0, 0), t, font=font_tick)[2:]
        draw.text((xp - tw / 2, mt + ph + 15), t, fill=(40, 40, 40), font=font_tick)

    palette = {
        "nnunet_crop": (33, 102, 172),
        "SAM2_prompt_2": (210, 85, 25),
        "SAM2_prompt_3": (53, 142, 56),
    }

    # Curves
    for model_name, vals in values_by_model.items():
        color = palette.get(model_name, (120, 120, 120))
        pts = []
        for i, yv in enumerate(vals, start=1):
            xp = x_to_px(i)
            yp = y_to_px(float(yv))
            if np.isfinite(yp):
                pts.append((xp, yp))
            else:
                if len(pts) >= 2:
                    draw.line(pts, fill=color, width=2)
                pts = []
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=4)

        # Draw points
        for i, yv in enumerate(vals, start=1):
            yp = y_to_px(float(yv))
            if not np.isfinite(yp):
                continue
            xp = x_to_px(i)
            r = 3
            draw.ellipse((xp - r, yp - r, xp + r, yp + r), fill=color, outline=color)

    # Title and axis labels
    title = f"{patient_id} - {metric_name}"
    tw, th = draw.textbbox((0, 0), title, font=font_title)[2:]
    draw.text(((width - tw) / 2, 16), title, fill=(20, 20, 20), font=font_title)

    xlabel = "Layer Index (from GT upper bound to lower bound)"
    xw, xh = draw.textbbox((0, 0), xlabel, font=font_axis)[2:]
    draw.text((ml + (pw - xw) / 2, mt + ph + 62), xlabel, fill=(20, 20, 20), font=font_axis)

    draw.text((18, mt - 10), metric_name, fill=(20, 20, 20), font=font_axis)

    # Legend (below the plot, centered)
    legend_items = list(values_by_model.keys())
    line_w = 56
    gap_item = 48
    text_w = [
        draw.textbbox((0, 0), name, font=font_legend)[2] - draw.textbbox((0, 0), name, font=font_legend)[0]
        for name in legend_items
    ]
    item_total_w = [line_w + 16 + w for w in text_w]
    total_w = sum(item_total_w) + gap_item * (len(item_total_w) - 1 if len(item_total_w) > 1 else 0)
    start_x = ml + (pw - total_w) / 2
    cy = height - 65

    cur_x = start_x
    for idx, model_name in enumerate(legend_items):
        color = palette.get(model_name, (120, 120, 120))
        draw.line((cur_x, cy, cur_x + line_w, cy), fill=color, width=6)
        draw.text((cur_x + line_w + 16, cy - 18), model_name, fill=(30, 30, 30), font=font_legend)
        cur_x += item_total_w[idx] + gap_item

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def fmt2(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.2f}"


def col_letter(col_idx_1based: int) -> str:
    n = col_idx_1based
    chars = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def make_cell_xml(r: int, c: int, value, style_idx: int) -> str:
    ref = f"{col_letter(c)}{r}"
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style_idx}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" s="{style_idx}" t="inlineStr"><is><t>{text}</t></is></c>'


def build_sheet_xml(table_rows: List[List]) -> str:
    rows_xml = []
    max_col = 10
    for r_idx, row in enumerate(table_rows, start=1):
        style_idx = 1 if r_idx <= 2 else 0
        cells_xml = []
        for c_idx in range(1, max_col + 1):
            v = row[c_idx - 1] if c_idx - 1 < len(row) else ""
            cell_xml = make_cell_xml(r_idx, c_idx, v, style_idx)
            if cell_xml:
                cells_xml.append(cell_xml)
        rows_xml.append(f'<row r="{r_idx}">' + "".join(cells_xml) + "</row>")

    merges = ["A1:A2", "B1:B2", "C1:C2", "D1:D2", "E1:G1", "H1:J1"]
    merge_xml = "".join([f'<mergeCell ref="{m}"/>' for m in merges])
    last_row = max(1, len(table_rows))

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:J{last_row}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<cols>'
        '<col min="1" max="1" width="12" customWidth="1"/>'
        '<col min="2" max="4" width="14" customWidth="1"/>'
        '<col min="5" max="10" width="18" customWidth="1"/>'
        '</cols>'
        '<sheetData>'
        + "".join(rows_xml)
        + '</sheetData>'
        f'<mergeCells count="{len(merges)}">{merge_xml}</mergeCells>'
        '</worksheet>'
    )


def write_xlsx(table_rows: List[List], out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="slice_metrics" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )

    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font>'
        '</fonts>'
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center"/>'
        '</xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    sheet = build_sheet_xml(table_rows)

    with zipfile.ZipFile(out_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


def resolve_prompt2_dir(path: Path) -> Path:
    if path.exists():
        return path
    fallback = path.parent
    if fallback.exists():
        return fallback
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-slice 2D Dice/HD95 within GT bounds and draw per-patient curves."
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(r"C:\Users\dell\Desktop\Eso-CTV\20260326\labelsTs"),
        help="GT folder (*.nii.gz)",
    )
    parser.add_argument(
        "--nnunet-dir",
        type=Path,
        default=Path(r"C:\Users\dell\Desktop\Eso-CTV\20260326\nnunet_crop"),
        help="nnunet_crop prediction folder",
    )
    parser.add_argument(
        "--sam2-prompt2-dir",
        type=Path,
        default=Path(r"C:\Users\dell\Desktop\Eso-CTV\20260326\Train\oracle_mask\mask_prompt_2\one_epoch"),
        help="SAM2 prompt_2 prediction folder",
    )
    parser.add_argument(
        "--sam2-prompt3-dir",
        type=Path,
        default=Path(r"C:\Users\dell\Desktop\Eso-CTV\20260326\Train\oracle_mask\mask_prompt_3\one_epoch"),
        help="SAM2 prompt_3 prediction folder",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(r"C:\Users\dell\Desktop\Eso-CTV\20260326\eval_outputs"),
        help="Output root folder",
    )
    parser.add_argument(
        "--tick-step",
        type=int,
        default=4,
        help="X-axis tick step for layer index (e.g., 4 -> 4,8,12,... plus final layer)",
    )
    args = parser.parse_args()

    args.sam2_prompt2_dir = resolve_prompt2_dir(args.sam2_prompt2_dir)

    pred_dirs = {
        "nnunet_crop": args.nnunet_dir,
        "SAM2_prompt_2": args.sam2_prompt2_dir,
        "SAM2_prompt_3": args.sam2_prompt3_dir,
    }
    method_order = ["nnunet_crop", "SAM2_prompt_2", "SAM2_prompt_3"]

    for n, d in [("gt", args.gt_dir), *pred_dirs.items()]:
        if not d.exists():
            raise FileNotFoundError(f"[{n}] directory not found: {d}")

    cases = collect_cases(args.gt_dir, pred_dirs)
    if not cases:
        raise RuntimeError("No shared patient files found across GT and all prediction folders.")

    out_dice = args.out_root / "dice2d_per_patient"
    out_hd95 = args.out_root / "hd95_2d_per_patient"
    out_xlsx = args.out_root / "slice_metrics_2d.xlsx"
    out_dice.mkdir(parents=True, exist_ok=True)
    out_hd95.mkdir(parents=True, exist_ok=True)

    header1 = [
        "Patient_ID",
        "Current_Z",
        "Lower_Bound_ID",
        "Upper_Bound_ID",
        "dice_2d",
        "",
        "",
        "hd95_2d_mm",
        "",
        "",
    ]
    header2 = [
        "",
        "",
        "",
        "",
        "nnunet_crop",
        "SAM2_prompt_2",
        "SAM2_prompt_3",
        "nnunet_crop",
        "SAM2_prompt_2",
        "SAM2_prompt_3",
    ]
    table_rows: List[List] = [header1, header2]

    print(f"[INFO] shared patients: {len(cases)}")
    print(f"[INFO] SAM2_prompt_2 dir used: {args.sam2_prompt2_dir}")

    for i, (pid, p) in enumerate(cases.items(), start=1):
        gt, gt_img = read_nii(p["gt"])
        gt_b = to_binary(gt)

        lower_id, upper_id = find_gt_bounds(gt_b)
        if lower_id < 0:
            print(f"[WARN] {pid}: GT empty, skipped")
            continue

        slice_ids = list(range(upper_id, lower_id - 1, -1))  # left->right: upper to lower

        # sitk spacing order is (x, y, z); 2D metrics use (x, y).
        sx, sy, _sz = gt_img.GetSpacing()
        spacing_xy = (float(sx), float(sy))

        pred_arrays: Dict[str, np.ndarray] = {}
        for name in method_order:
            arr, _ = read_nii(p[name])
            pred_arrays[name] = to_binary(arr)

        dice_vals: Dict[str, List[float]] = {k: [] for k in method_order}
        hd95_vals: Dict[str, List[float]] = {k: [] for k in method_order}

        for z in slice_ids:
            gt2d = gt_b[z]
            for name in method_order:
                arr = pred_arrays[name]
                pred2d = arr[z]
                dice_vals[name].append(dice_2d(pred2d, gt2d))
                hd95_vals[name].append(hd95_2d(pred2d, gt2d, spacing_xy))

            row = [pid, int(z), lower_id, upper_id]
            row.extend([fmt2(dice_vals[m][-1]) for m in method_order])
            row.extend([fmt2(hd95_vals[m][-1]) for m in method_order])
            table_rows.append(row)

        plot_per_patient(
            patient_id=pid,
            metric_name="2D Dice",
            values_by_model=dice_vals,
            out_path=out_dice / f"{pid}.png",
            tick_step=args.tick_step,
        )
        plot_per_patient(
            patient_id=pid,
            metric_name="2D HD95 (mm)",
            values_by_model=hd95_vals,
            out_path=out_hd95 / f"{pid}.png",
            tick_step=args.tick_step,
        )

        print(
            f"[DONE] {i:>3}/{len(cases)} {pid} | "
            f"GT bounds: lower={lower_id}, upper={upper_id}, layers={len(slice_ids)}"
        )

    print(f"[OK] Dice plots: {out_dice}")
    print(f"[OK] HD95 plots: {out_hd95}")
    write_xlsx(table_rows, out_xlsx)
    print(f"[OK] Slice metrics xlsx: {out_xlsx}")


if __name__ == "__main__":
    main()
