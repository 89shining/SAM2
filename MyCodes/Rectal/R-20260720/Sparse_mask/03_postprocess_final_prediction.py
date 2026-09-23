#!/usr/bin/env python3
"""Postprocess final predictions using axial-slice-only 2D operations.

Per axial slice, the exact order is:
  1. retain the largest 8-connected component;
  2. erode with a radius-2 disk;
  3. retain the largest 8-connected component;
  4. dilate with a radius-2 disk;
  5. fill enclosed holes whose area is smaller than 16 pixels;
  6. Gaussian smooth with sigma=1.5 (in-plane only);
  7. threshold at 0.5;
  8. retain the largest 8-connected component.

No operation connects or smooths voxels between different slices.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


DEFAULT_PATIENT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask/test"
)
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-root", type=Path, default=DEFAULT_PATIENT_ROOT)
    parser.add_argument("--input-name", default="final_predict_raw.nii.gz")
    parser.add_argument("--output-name", default="final_predict.nii.gz")
    parser.add_argument("--backup-name", default="final_predict_backup.nii.gz")
    parser.add_argument("--morph-radius", type=int, default=2)
    parser.add_argument(
        "--hole-area",
        type=int,
        default=16,
        help="Fill enclosed holes with area strictly smaller than this pixel count.",
    )
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output and backup files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up an existing output before replacing it.",
    )
    return parser.parse_args()


def patient_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Patient root does not exist: {root}")
    direct = sorted(path for path in root.glob("p_*") if path.is_dir())
    if direct:
        return direct
    nested = sorted(path for path in root.glob("*/p_*") if path.is_dir())
    if not nested:
        raise RuntimeError(f"No p_* patient folders found under: {root}")
    return nested


def disk(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("--morph-radius must be >= 0")
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def keep_largest_8(mask_yx: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask_yx > 0, structure=CONNECTIVITY_8)
    if count == 0:
        return np.zeros_like(mask_yx, dtype=bool)
    areas = np.bincount(labels.ravel())
    largest_label = int(np.argmax(areas[1:]) + 1)
    return labels == largest_label


def fill_small_enclosed_holes(mask_yx: np.ndarray, area_threshold: int) -> np.ndarray:
    """Fill 8-connected background components not touching the image border."""
    if area_threshold <= 0:
        return mask_yx.astype(bool, copy=True)
    background = ~mask_yx.astype(bool)
    labels, count = ndimage.label(background, structure=CONNECTIVITY_8)
    if count == 0:
        return mask_yx.astype(bool, copy=True)

    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    areas = np.bincount(labels.ravel())
    enclosed = np.ones(count + 1, dtype=bool)
    enclosed[0] = False
    enclosed[border_labels] = False
    small_hole_labels = np.flatnonzero(enclosed & (areas < area_threshold))
    return mask_yx.astype(bool) | np.isin(labels, small_hole_labels)


def process_slice(
    mask_yx: np.ndarray,
    morph_disk: np.ndarray,
    hole_area: int,
    sigma: float,
    threshold: float,
) -> np.ndarray:
    current = keep_largest_8(mask_yx)
    if not current.any():
        return current

    eroded = ndimage.binary_erosion(current, structure=morph_disk, border_value=0)
    eroded = keep_largest_8(eroded)
    if not eroded.any():
        # A very small valid object can disappear after erosion. Keep the first
        # largest-component result instead of deleting the complete slice.
        restored = current
    else:
        restored = ndimage.binary_dilation(
            eroded, structure=morph_disk, border_value=0
        )

    filled = fill_small_enclosed_holes(restored, hole_area)
    smoothed = ndimage.gaussian_filter(
        filled.astype(np.float32), sigma=sigma, mode="constant", cval=0.0
    )
    binary = smoothed >= threshold
    return keep_largest_8(binary)


def process_volume(
    mask_zyx: np.ndarray,
    morph_radius: int,
    hole_area: int,
    sigma: float,
    threshold: float,
) -> np.ndarray:
    if hole_area < 0:
        raise ValueError("--hole-area must be >= 0")
    if sigma < 0:
        raise ValueError("--sigma must be >= 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    morph_disk = disk(morph_radius)
    output = np.zeros_like(mask_zyx, dtype=bool)
    for z, mask_yx in enumerate(mask_zyx > 0):
        output[z] = process_slice(
            mask_yx, morph_disk, hole_area, sigma, threshold
        )
    return output


def write_like(mask_zyx: np.ndarray, reference: sitk.Image, path: Path) -> None:
    output = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path), useCompression=True)


def main() -> None:
    args = parse_args()
    patients = patient_dirs(args.patient_root)

    jobs: list[tuple[Path, Path, Path]] = []
    for patient in patients:
        source = patient / args.input_name
        output = patient / args.output_name
        backup = patient / args.backup_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing input: {source}")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        if (
            output.exists()
            and not args.no_backup
            and backup.exists()
            and not args.overwrite
        ):
            raise FileExistsError(f"Backup exists; use --overwrite: {backup}")
        jobs.append((source, output, backup))

    for source, output, backup in jobs:
        reference = sitk.ReadImage(str(source))
        raw = sitk.GetArrayFromImage(reference) > 0
        processed = process_volume(
            raw,
            args.morph_radius,
            args.hole_area,
            args.sigma,
            args.threshold,
        )

        if output.exists() and not args.no_backup:
            shutil.copy2(output, backup)
        write_like(processed, reference, output)
        print(
            f"{source.parent.name}: voxels={int(raw.sum())}->{int(processed.sum())}, "
            f"nonempty_slices={int(raw.any(axis=(1, 2)).sum())}->"
            f"{int(processed.any(axis=(1, 2)).sum())} -> {output.name}"
        )

    print(
        f"Done: {len(jobs)} patients; largest8 -> erode{args.morph_radius} -> "
        f"largest8 -> dilate{args.morph_radius} -> fill holes <{args.hole_area}px -> "
        f"Gaussian sigma={args.sigma:g} -> threshold={args.threshold:g} -> largest8"
    )


if __name__ == "__main__":
    main()
