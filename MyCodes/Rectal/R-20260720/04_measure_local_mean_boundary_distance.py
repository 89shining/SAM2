#!/usr/bin/env python3
"""Measure local mean boundary distance for every processed 2D error component."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


DEFAULT_PROMPT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For every axial-slice component, compute mean distance from the "
            "error-component boundary to the corresponding reference contour."
        )
    )
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--input-tag",
        default="slice_top3_min50mm2",
        help="Read pos_<tag>.nii.gz and neg_<tag>.nii.gz.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="2D connectivity for error components (default: 8).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV; default is under <prompt-root>.",
    )
    return parser.parse_args()


def patient_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("p_")), path.name
    except ValueError:
        return 10**12, path.name


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def binary_boundary_2d(mask_yx: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner boundary using 8-neighbour morphology."""
    mask = mask_yx > 0
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    return mask & ~eroded


def component_rows_for_mask(
    patient: str,
    prompt_type: str,
    error_zyx: np.ndarray,
    reference_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    connectivity: int,
) -> list[dict[str, object]]:
    """Measure FN-to-prediction or FP-to-GT contour distances per 2D component."""
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    sampling_yx = (float(spacing_xyz[1]), float(spacing_xyz[0]))
    pixel_area_mm2 = float(spacing_xyz[0] * spacing_xyz[1])
    rows: list[dict[str, object]] = []

    for z, error_yx in enumerate(error_zyx):
        labels, count = ndimage.label(error_yx > 0, structure=structure)
        if count == 0:
            continue

        reference_boundary = binary_boundary_2d(reference_zyx[z])
        if reference_boundary.any():
            # At each pixel, distance to the nearest reference-contour pixel in mm.
            distance_to_reference = ndimage.distance_transform_edt(
                ~reference_boundary,
                sampling=sampling_yx,
            )
        else:
            distance_to_reference = None

        component_areas = np.bincount(labels.ravel())
        order = sorted(
            range(1, count + 1),
            key=lambda label: (-int(component_areas[label]), label),
        )
        for rank, label_value in enumerate(order, start=1):
            component = labels == label_value
            component_boundary = binary_boundary_2d(component)
            boundary_pixels = int(component_boundary.sum())
            if distance_to_reference is None:
                mean_distance_mm = math.nan
                min_distance_mm = math.nan
                max_distance_mm = math.nan
                status = "reference_contour_empty"
            else:
                distances = distance_to_reference[component_boundary]
                mean_distance_mm = float(distances.mean())
                min_distance_mm = float(distances.min())
                max_distance_mm = float(distances.max())
                status = "ok"

            rows.append(
                {
                    "patient": patient,
                    "prompt_type": prompt_type,
                    "error_type": "FN" if prompt_type == "pos" else "FP",
                    "reference_contour": (
                        "nnunet_prediction" if prompt_type == "pos" else "GT"
                    ),
                    "slice_index_z": z,
                    "component_rank_by_area": rank,
                    "component_label": label_value,
                    "component_area_pixels": int(component_areas[label_value]),
                    "component_area_mm2": (
                        int(component_areas[label_value]) * pixel_area_mm2
                    ),
                    "component_boundary_pixels": boundary_pixels,
                    "mean_boundary_distance_mm": mean_distance_mm,
                    "min_boundary_distance_mm": min_distance_mm,
                    "max_boundary_distance_mm": max_distance_mm,
                    "status": status,
                    "spacing_x_mm": float(spacing_xyz[0]),
                    "spacing_y_mm": float(spacing_xyz[1]),
                    "spacing_z_mm": float(spacing_xyz[2]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if not args.input_tag or any(c in args.input_tag for c in "/\\"):
        raise ValueError("--input-tag must be a non-empty filename-safe tag")

    test_dir = args.prompt_root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    patients = sorted(
        (p for p in test_dir.glob("p_*") if p.is_dir()),
        key=patient_sort_key,
    )
    if not patients:
        raise RuntimeError(f"No patient folders found in: {test_dir}")

    output_csv = args.output_csv or (
        args.prompt_root
        / f"test_{args.input_tag}_local_mean_boundary_distance.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for patient_dir in patients:
        paths = {
            "pos": patient_dir / f"pos_{args.input_tag}.nii.gz",
            "neg": patient_dir / f"neg_{args.input_tag}.nii.gz",
            "prediction": patient_dir / "nnunet.nii.gz",
            "gt": patient_dir / "CTV.nii.gz",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}")

        images = {name: sitk.ReadImage(str(path)) for name, path in paths.items()}
        reference_image = images["gt"]
        for name, image in images.items():
            if not same_geometry(reference_image, image):
                raise ValueError(
                    f"Geometry mismatch for {name} in patient: {patient_dir}"
                )

        arrays = {
            name: sitk.GetArrayFromImage(image) > 0
            for name, image in images.items()
        }
        pos_rows = component_rows_for_mask(
            patient_dir.name,
            "pos",
            arrays["pos"],
            arrays["prediction"],
            reference_image.GetSpacing(),
            args.connectivity,
        )
        neg_rows = component_rows_for_mask(
            patient_dir.name,
            "neg",
            arrays["neg"],
            arrays["gt"],
            reference_image.GetSpacing(),
            args.connectivity,
        )
        all_rows.extend(pos_rows)
        all_rows.extend(neg_rows)
        pos_valid = sum(row["status"] == "ok" for row in pos_rows)
        neg_valid = sum(row["status"] == "ok" for row in neg_rows)
        print(
            f"test/{patient_dir.name}: "
            f"pos components={len(pos_rows)} (valid={pos_valid}), "
            f"neg components={len(neg_rows)} (valid={neg_valid})"
        )

    fieldnames = [
        "patient",
        "prompt_type",
        "error_type",
        "reference_contour",
        "slice_index_z",
        "component_rank_by_area",
        "component_label",
        "component_area_pixels",
        "component_area_mm2",
        "component_boundary_pixels",
        "mean_boundary_distance_mm",
        "min_boundary_distance_mm",
        "max_boundary_distance_mm",
        "status",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    valid_rows = [row for row in all_rows if row["status"] == "ok"]
    invalid_count = len(all_rows) - len(valid_rows)
    print(
        f"Done: patients={len(patients)}, components={len(all_rows)}, "
        f"valid={len(valid_rows)}, reference-empty={invalid_count}\n"
        f"Saved: {output_csv}"
    )


if __name__ == "__main__":
    main()
