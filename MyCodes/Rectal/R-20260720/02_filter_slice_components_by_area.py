#!/usr/bin/env python3
"""Filter per-slice 2D components in processed test prompts by physical area."""

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
        description="Remove small 2D components from processed test pos/neg masks."
    )
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--input-tag",
        default="slice_top3",
        help="Read pos_<tag>.nii.gz and neg_<tag>.nii.gz (default: slice_top3).",
    )
    parser.add_argument(
        "--min-area-mm2",
        type=float,
        default=50.0,
        help="Delete components with physical area below this value (default: 50).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="2D connectivity (default: 8).",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help=(
            "Output suffix. Default: <input-tag>_min<area>mm2, "
            "for example slice_top3_min50mm2."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def filter_mask(
    mask_zyx: np.ndarray,
    pixel_area_mm2: float,
    min_area_mm2: float,
    connectivity: int,
) -> tuple[np.ndarray, int, int, int]:
    """Filter each axial slice independently and return component statistics."""
    output = np.zeros_like(mask_zyx, dtype=bool)
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    components_before = 0
    components_kept = 0
    components_removed = 0

    for z, source_yx in enumerate(mask_zyx):
        labels, count = ndimage.label(source_yx > 0, structure=structure)
        components_before += int(count)
        if count == 0:
            continue

        pixel_counts = np.bincount(labels.ravel())
        areas_mm2 = pixel_counts.astype(np.float64) * pixel_area_mm2
        keep_label = areas_mm2 >= min_area_mm2
        keep_label[0] = False
        output[z] = keep_label[labels]

        kept = int(np.count_nonzero(keep_label[1:]))
        components_kept += kept
        components_removed += int(count) - kept

    return output, components_before, components_kept, components_removed


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
    if args.min_area_mm2 < 0:
        raise ValueError("--min-area-mm2 must be >= 0")
    if not args.input_tag or any(c in args.input_tag for c in "/\\"):
        raise ValueError("--input-tag must be a non-empty filename-safe tag")

    output_tag = args.output_tag or (
        f"{args.input_tag}_min{number_tag(args.min_area_mm2)}mm2"
    )
    if not output_tag or any(c in output_tag for c in "/\\"):
        raise ValueError("--output-tag must be a non-empty filename-safe tag")

    patients = get_test_patients(args.prompt_root)
    jobs: list[tuple[Path, Path, Path, Path]] = []

    # Validate all source files and output conflicts before writing.
    for patient in patients:
        pos_input = patient / f"pos_{args.input_tag}.nii.gz"
        neg_input = patient / f"neg_{args.input_tag}.nii.gz"
        pos_output = patient / f"pos_{output_tag}.nii.gz"
        neg_output = patient / f"neg_{output_tag}.nii.gz"
        for source in (pos_input, neg_input):
            if not source.is_file():
                raise FileNotFoundError(f"Missing processed prompt: {source}")
        for output in (pos_output, neg_output):
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {output}")
        jobs.append((pos_input, neg_input, pos_output, neg_output))

    totals = {
        "pos_before": 0,
        "pos_kept": 0,
        "pos_removed": 0,
        "neg_before": 0,
        "neg_kept": 0,
        "neg_removed": 0,
    }

    for pos_input, neg_input, pos_output, neg_output in jobs:
        pos_image = sitk.ReadImage(str(pos_input))
        neg_image = sitk.ReadImage(str(neg_input))
        if not same_geometry(pos_image, neg_image):
            raise ValueError(f"Positive/negative geometry mismatch: {pos_input.parent}")

        pos_spacing = pos_image.GetSpacing()
        neg_spacing = neg_image.GetSpacing()
        pos_result = filter_mask(
            sitk.GetArrayFromImage(pos_image) > 0,
            float(pos_spacing[0] * pos_spacing[1]),
            args.min_area_mm2,
            args.connectivity,
        )
        neg_result = filter_mask(
            sitk.GetArrayFromImage(neg_image) > 0,
            float(neg_spacing[0] * neg_spacing[1]),
            args.min_area_mm2,
            args.connectivity,
        )
        pos_mask, pos_before, pos_kept, pos_removed = pos_result
        neg_mask, neg_before, neg_kept, neg_removed = neg_result

        if np.any(pos_mask & neg_mask):
            raise ValueError(f"Filtered positive/negative masks overlap: {pos_input.parent}")

        write_like(pos_mask, pos_image, pos_output)
        write_like(neg_mask, neg_image, neg_output)

        totals["pos_before"] += pos_before
        totals["pos_kept"] += pos_kept
        totals["pos_removed"] += pos_removed
        totals["neg_before"] += neg_before
        totals["neg_kept"] += neg_kept
        totals["neg_removed"] += neg_removed
        print(
            f"test/{pos_input.parent.name}: "
            f"pos={pos_before}->{pos_kept} (removed={pos_removed}), "
            f"neg={neg_before}->{neg_kept} (removed={neg_removed})"
        )

    print(
        f"Done: {len(patients)} test patients; threshold={args.min_area_mm2:g} mm2; "
        f"pos={totals['pos_before']}->{totals['pos_kept']} "
        f"(removed={totals['pos_removed']}), "
        f"neg={totals['neg_before']}->{totals['neg_kept']} "
        f"(removed={totals['neg_removed']}); "
        f"outputs=pos_{output_tag}.nii.gz/neg_{output_tag}.nii.gz"
    )


if __name__ == "__main__":
    main()
