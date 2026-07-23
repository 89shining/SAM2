#!/usr/bin/env python3
"""Keep the largest 2D components independently on every axial slice."""

from __future__ import annotations

import argparse
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
            "For each test patient and each axial slice, retain the largest "
            "2D components independently in pos_raw and neg_raw."
        )
    )
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of largest components retained per slice and prompt type (default: 3).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="2D pixel connectivity (default: 8).",
    )
    parser.add_argument(
        "--output-tag",
        default="slice_top3",
        help="Output suffix, producing pos_<tag>.nii.gz and neg_<tag>.nii.gz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def keep_slice_topk(
    mask_zyx: np.ndarray, top_k: int, connectivity: int
) -> tuple[np.ndarray, int, int]:
    """Return filtered [Z,Y,X] mask and component counts before/after."""
    output = np.zeros_like(mask_zyx, dtype=bool)
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    total_before = 0
    total_after = 0

    for z, source_yx in enumerate(mask_zyx):
        labels, count = ndimage.label(source_yx > 0, structure=structure)
        total_before += int(count)
        if count == 0:
            continue

        areas = np.bincount(labels.ravel())
        component_labels = np.arange(1, count + 1)
        # Stable ordering makes equal-area ties deterministic by label number.
        order = np.argsort(-areas[1:], kind="stable")
        kept_labels = component_labels[order[:top_k]]
        output[z] = np.isin(labels, kept_labels)
        total_after += len(kept_labels)

    return output, total_before, total_after


def write_like(mask_zyx: np.ndarray, reference: sitk.Image, output: Path) -> None:
    image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(output), useCompression=True)


def get_test_patients(prompt_root: Path) -> list[Path]:
    test_dir = prompt_root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    patients = sorted(p for p in test_dir.glob("p_*") if p.is_dir())
    if not patients:
        raise RuntimeError(f"No p_* patient folders found in: {test_dir}")
    return patients


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not args.output_tag or any(c in args.output_tag for c in "/\\"):
        raise ValueError("--output-tag must be a non-empty filename-safe tag")

    patients = get_test_patients(args.prompt_root)
    jobs: list[tuple[Path, Path, Path, Path]] = []

    # Validate all inputs and conflicts before creating any output.
    for patient in patients:
        pos_raw = patient / "pos_raw.nii.gz"
        neg_raw = patient / "neg_raw.nii.gz"
        pos_output = patient / f"pos_{args.output_tag}.nii.gz"
        neg_output = patient / f"neg_{args.output_tag}.nii.gz"

        for source in (pos_raw, neg_raw):
            if not source.is_file():
                raise FileNotFoundError(f"Missing raw prompt: {source}")
        for output in (pos_output, neg_output):
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {output}")
        jobs.append((pos_raw, neg_raw, pos_output, neg_output))

    total_before = {"pos": 0, "neg": 0}
    total_after = {"pos": 0, "neg": 0}
    total_voxels = {"pos": 0, "neg": 0}

    for pos_path, neg_path, pos_output, neg_output in jobs:
        pos_image = sitk.ReadImage(str(pos_path))
        neg_image = sitk.ReadImage(str(neg_path))
        if not same_geometry(pos_image, neg_image):
            raise ValueError(f"Positive/negative geometry mismatch: {pos_path.parent}")

        pos_raw = sitk.GetArrayFromImage(pos_image) > 0
        neg_raw = sitk.GetArrayFromImage(neg_image) > 0
        pos, pos_before, pos_after = keep_slice_topk(
            pos_raw, args.top_k, args.connectivity
        )
        neg, neg_before, neg_after = keep_slice_topk(
            neg_raw, args.top_k, args.connectivity
        )

        # Raw positive and negative errors should already be disjoint.
        overlap = pos & neg
        if overlap.any():
            raise ValueError(
                f"Filtered positive/negative masks overlap in: {pos_path.parent}"
            )

        write_like(pos, pos_image, pos_output)
        write_like(neg, neg_image, neg_output)

        total_before["pos"] += pos_before
        total_before["neg"] += neg_before
        total_after["pos"] += pos_after
        total_after["neg"] += neg_after
        total_voxels["pos"] += int(pos.sum())
        total_voxels["neg"] += int(neg.sum())
        print(
            f"test/{pos_path.parent.name}: "
            f"pos components={pos_before}->{pos_after}, voxels={int(pos.sum())}; "
            f"neg components={neg_before}->{neg_after}, voxels={int(neg.sum())}"
        )

    print(
        f"Done: {len(patients)} test patients; top_k={args.top_k}, "
        f"connectivity={args.connectivity}; "
        f"pos components={total_before['pos']}->{total_after['pos']}, "
        f"voxels={total_voxels['pos']}; "
        f"neg components={total_before['neg']}->{total_after['neg']}, "
        f"voxels={total_voxels['neg']}"
    )


if __name__ == "__main__":
    main()
