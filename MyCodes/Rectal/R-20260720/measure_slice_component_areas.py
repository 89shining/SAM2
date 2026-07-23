#!/usr/bin/env python3
"""Measure 2D connected-component areas in processed test prompt masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


DEFAULT_PROMPT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-slice component areas in processed pos/neg masks."
    )
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--input-tag",
        default="slice_top3",
        help="Read pos_<tag>.nii.gz and neg_<tag>.nii.gz (default: slice_top3).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="2D connectivity used for measurement (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV directory; default is <prompt-root>.",
    )
    parser.add_argument(
        "--include-empty-slices",
        action="store_true",
        help="Include slices without foreground in the slice-summary CSV.",
    )
    return parser.parse_args()


def patient_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("p_")), path.name
    except ValueError:
        return 10**12, path.name


def measure_mask(
    patient: str,
    prompt_type: str,
    image: sitk.Image,
    connectivity: int,
    include_empty: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mask_zyx = sitk.GetArrayFromImage(image) > 0
    spacing_x, spacing_y, spacing_z = image.GetSpacing()
    pixel_area_mm2 = float(spacing_x * spacing_y)
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )

    component_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []

    for z, mask_yx in enumerate(mask_zyx):
        labels, count = ndimage.label(mask_yx, structure=structure)
        areas = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=int)
        sorted_areas = sorted((int(v) for v in areas), reverse=True)

        for rank, area_pixels in enumerate(sorted_areas, start=1):
            component_rows.append(
                {
                    "patient": patient,
                    "prompt_type": prompt_type,
                    "slice_index_z": z,
                    "component_rank": rank,
                    "area_pixels": area_pixels,
                    "area_mm2": area_pixels * pixel_area_mm2,
                    "spacing_x_mm": spacing_x,
                    "spacing_y_mm": spacing_y,
                    "spacing_z_mm": spacing_z,
                }
            )

        if count or include_empty:
            slice_rows.append(
                {
                    "patient": patient,
                    "prompt_type": prompt_type,
                    "slice_index_z": z,
                    "component_count": int(count),
                    "total_area_pixels": int(sum(sorted_areas)),
                    "total_area_mm2": float(sum(sorted_areas) * pixel_area_mm2),
                    "largest_area_pixels": sorted_areas[0] if sorted_areas else 0,
                    "largest_area_mm2": (
                        sorted_areas[0] * pixel_area_mm2 if sorted_areas else 0.0
                    ),
                }
            )

    return component_rows, slice_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.input_tag or any(c in args.input_tag for c in "/\\"):
        raise ValueError("--input-tag must be a non-empty filename-safe tag")

    test_dir = args.prompt_root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    patients = sorted(
        (p for p in test_dir.glob("p_*") if p.is_dir()), key=patient_sort_key
    )
    if not patients:
        raise RuntimeError(f"No patient folders found in: {test_dir}")

    output_dir = args.output_dir or args.prompt_root
    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []

    for patient_dir in patients:
        patient_component_count = {"pos": 0, "neg": 0}
        for prompt_type in ("pos", "neg"):
            input_path = patient_dir / f"{prompt_type}_{args.input_tag}.nii.gz"
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing processed prompt: {input_path}")
            image = sitk.ReadImage(str(input_path))
            details, summaries = measure_mask(
                patient_dir.name,
                prompt_type,
                image,
                args.connectivity,
                args.include_empty_slices,
            )
            component_rows.extend(details)
            slice_rows.extend(summaries)
            patient_component_count[prompt_type] = len(details)

        print(
            f"test/{patient_dir.name}: "
            f"pos 2D components={patient_component_count['pos']}, "
            f"neg 2D components={patient_component_count['neg']}"
        )

    component_path = output_dir / f"test_{args.input_tag}_all_component_areas.csv"
    summary_path = output_dir / f"test_{args.input_tag}_slice_area_summary.csv"
    write_csv(
        component_path,
        component_rows,
        [
            "patient",
            "prompt_type",
            "slice_index_z",
            "component_rank",
            "area_pixels",
            "area_mm2",
            "spacing_x_mm",
            "spacing_y_mm",
            "spacing_z_mm",
        ],
    )
    write_csv(
        summary_path,
        slice_rows,
        [
            "patient",
            "prompt_type",
            "slice_index_z",
            "component_count",
            "total_area_pixels",
            "total_area_mm2",
            "largest_area_pixels",
            "largest_area_mm2",
        ],
    )
    print(
        f"Done: patients={len(patients)}, components={len(component_rows)}\n"
        f"Component details: {component_path}\nSlice summaries: {summary_path}"
    )


if __name__ == "__main__":
    main()
