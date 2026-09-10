#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Offline preprocessing for the T-20260909 ESO CTV Stage-1 experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_RAW_DATA_ROOT = Path(
    "/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260905_CTV/datanii"
)
DEFAULT_DATA_ROOT = Path(
    "/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/PreprocessDataNii"
)
DEFAULT_TARGET_Z_SPACING = 5.0


def patient_sort_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def window_to_float01(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Clip HU by WC/WW and normalize directly to float32 [0, 1]."""
    image = image.astype(np.float32)
    lower = float(window_center) - float(window_width) / 2.0
    upper = float(window_center) + float(window_width) / 2.0
    return ((np.clip(image, lower, upper) - lower) / (upper - lower)).astype(np.float32)


def make_z_resample_reference(
    image: sitk.Image,
    target_z_spacing: float = DEFAULT_TARGET_Z_SPACING,
) -> sitk.Image:
    """Change only Z spacing/size and preserve the original XY grid."""
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_z = int(
        round((old_size[2] - 1) * old_spacing[2] / float(target_z_spacing))
    ) + 1
    reference = sitk.Image(
        [int(old_size[0]), int(old_size[1]), max(1, new_z)], sitk.sitkFloat32
    )
    reference.SetSpacing(
        (float(old_spacing[0]), float(old_spacing[1]), float(target_z_spacing))
    )
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    return reference


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    is_mask: bool,
    output_pixel_type,
) -> sitk.Image:
    interpolation = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    default_value = 0.0 if is_mask else -1000.0
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolation,
        default_value,
        output_pixel_type,
    )


def _write_volume(array_zyx: np.ndarray, reference: sitk.Image, path: Path, pixel_type) -> sitk.Image:
    output = sitk.GetImageFromArray(array_zyx)
    output.CopyInformation(reference)
    output = sitk.Cast(output, pixel_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(path), True)
    return output


def preprocess_case(
    source_dir: Path,
    destination_dir: Path,
    target_z_spacing: float,
    image_size: int,
    window_center: float,
    window_width: float,
    overwrite: bool,
) -> dict:
    image_out = destination_dir / "image.nii.gz"
    ctv_out = destination_dir / "CTV.nii.gz"
    if (image_out.exists() or ctv_out.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists for {source_dir.name}; use --overwrite to replace it"
        )

    original_image = sitk.ReadImage(str(source_dir / "image.nii.gz"))
    original_ctv = sitk.ReadImage(str(source_dir / "CTV.nii.gz"))
    if original_image.GetSize()[:2] != (int(image_size), int(image_size)):
        raise ValueError(
            f"{source_dir.name}: expected original XY size {image_size}x{image_size}, "
            f"got {original_image.GetSize()[:2]}"
        )
    reference = make_z_resample_reference(original_image, target_z_spacing)
    image = sitk.GetArrayFromImage(
        resample_to_reference(
            original_image, reference, is_mask=False, output_pixel_type=sitk.sitkFloat32
        )
    ).astype(np.float32)
    ctv = sitk.GetArrayFromImage(
        resample_to_reference(
            original_ctv, reference, is_mask=True, output_pixel_type=sitk.sitkUInt8
        )
    )
    ctv = (ctv > 0).astype(np.uint8)
    normalized = window_to_float01(image, window_center, window_width)
    image_512 = normalized.astype(np.float32)
    ctv_512 = (ctv > 0).astype(np.uint8)
    assert image.shape == ctv.shape
    assert image_512.shape == ctv_512.shape
    assert image_512.shape[1:] == (int(image_size), int(image_size))
    assert np.isfinite(image_512).all()
    assert set(np.unique(ctv_512).tolist()).issubset({0, 1})

    saved_image = _write_volume(image_512, reference, image_out, sitk.sitkFloat32)
    saved_ctv = _write_volume(ctv_512, reference, ctv_out, sitk.sitkUInt8)
    if any(
        (
            saved_image.GetSize() != saved_ctv.GetSize(),
            saved_image.GetSpacing() != saved_ctv.GetSpacing(),
            saved_image.GetOrigin() != saved_ctv.GetOrigin(),
            saved_image.GetDirection() != saved_ctv.GetDirection(),
        )
    ):
        raise RuntimeError(f"Saved CT/CTV geometry mismatch for {source_dir.name}")
    return {
        "patient": source_dir.name,
        "source_size_xyz": list(original_image.GetSize()),
        "source_spacing_xyz": list(original_image.GetSpacing()),
        "source_origin_xyz": list(original_image.GetOrigin()),
        "source_direction": list(original_image.GetDirection()),
        "saved_size_zyx": list(image_512.shape),
        "saved_size_xyz": list(saved_image.GetSize()),
        "saved_spacing_xyz": list(saved_image.GetSpacing()),
        "saved_origin_xyz": list(saved_image.GetOrigin()),
        "saved_direction": list(saved_image.GetDirection()),
        "saved_image_range": [float(image_512.min()), float(image_512.max())],
        "ctv_positive_voxels": int(ctv_512.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Offline preprocessing for ESO CTV Stage-1")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_RAW_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--target-z-spacing", type=float, default=DEFAULT_TARGET_Z_SPACING)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_partition = args.source_root / args.partition
    patients = sorted(source_partition.glob("p_*"), key=patient_sort_key)
    if not patients:
        raise FileNotFoundError(f"No p_* patients found in {source_partition}")
    records = []
    for index, source_dir in enumerate(patients, 1):
        if not (source_dir / "image.nii.gz").is_file() or not (source_dir / "CTV.nii.gz").is_file():
            raise FileNotFoundError(f"Missing image.nii.gz or CTV.nii.gz in {source_dir}")
        record = preprocess_case(
            source_dir,
            args.output_root / args.partition / source_dir.name,
            args.target_z_spacing,
            args.image_size,
            args.window_center,
            args.window_width,
            args.overwrite,
        )
        records.append(record)
        print(f"[{index}/{len(patients)}] {source_dir.name} -> {record['saved_size_zyx']}", flush=True)

    manifest = {
        "schema_version": 2,
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "partition": args.partition,
        "target_z_spacing": float(args.target_z_spacing),
        "window_center": args.window_center,
        "window_width": args.window_width,
        "normalized_range": [0.0, 1.0],
        "image_size": args.image_size,
        "image_interpolation": "SimpleITK linear to shared final reference",
        "ctv_interpolation": "SimpleITK nearest to shared final reference",
        "patients": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"preprocess_manifest_{args.partition}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
