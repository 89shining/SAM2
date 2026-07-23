#!/usr/bin/env python3
"""Erode raw prompts, keep slice Top-K, then filter by physical area."""

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
            "Apply axial-slice 2D erosion to test raw prompts, then retain "
            "the largest components and filter them by physical area."
        )
    )
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--erosion-radius",
        type=int,
        default=0,
        help="In-plane erosion radius in pixels; 0 keeps the raw mask unchanged.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Keep the largest K components per slice and prompt type (default: 3).",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="2D connectivity used after erosion (default: 8).",
    )
    parser.add_argument(
        "--min-area-mm2",
        type=float,
        default=50.0,
        help="Delete retained components below this physical area (default: 50).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs with the same radius.",
    )
    return parser.parse_args()


def erode_and_keep_topk(
    mask_zyx: np.ndarray,
    radius: int,
    top_k: int,
    connectivity: int,
    pixel_area_mm2: float,
    min_area_mm2: float,
) -> tuple[np.ndarray, int, int, int]:
    """Erode, retain Top-K, and area-filter components on every slice."""
    mask = mask_zyx > 0
    if radius > 0:
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        disk = (xx * xx + yy * yy) <= radius * radius
    else:
        disk = None
    component_structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    output = np.zeros_like(mask, dtype=bool)
    components_before = 0
    components_topk = 0
    components_area_kept = 0

    for z, mask_yx in enumerate(mask):
        current = mask_yx
        if disk is not None:
            current = ndimage.binary_erosion(
                current,
                structure=disk,
                border_value=0,
            )

        labels, count = ndimage.label(current, structure=component_structure)
        components_before += int(count)
        if count == 0:
            continue
        areas = np.bincount(labels.ravel())
        component_labels = np.arange(1, count + 1)
        order = np.argsort(-areas[1:], kind="stable")
        kept_labels = component_labels[order[:top_k]]
        components_topk += len(kept_labels)
        qualifying_labels = [
            int(label)
            for label in kept_labels
            if float(areas[label] * pixel_area_mm2) >= min_area_mm2
        ]
        output[z] = np.isin(labels, qualifying_labels)
        components_area_kept += len(qualifying_labels)

    return output, components_before, components_topk, components_area_kept


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
    if args.erosion_radius < 0:
        raise ValueError("--erosion-radius must be >= 0")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.min_area_mm2 < 0:
        raise ValueError("--min-area-mm2 must be >= 0")

    patients = get_test_patients(args.prompt_root)
    area_tag = f"{args.min_area_mm2:g}".replace(".", "p")
    jobs: list[tuple[Path, Path]] = []
    for patient in patients:
        for prompt_type in ("pos", "neg"):
            source = patient / f"{prompt_type}_raw.nii.gz"
            output = (
                patient
                / (
                    f"{prompt_type}_erode{args.erosion_radius}_top{args.top_k}"
                    f"_min{area_tag}mm2.nii.gz"
                )
            )
            if not source.is_file():
                raise FileNotFoundError(f"Missing raw prompt: {source}")
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {output}")
            jobs.append((source, output))

    total_before = {"pos": 0, "neg": 0}
    total_after = {"pos": 0, "neg": 0}
    total_components_before = {"pos": 0, "neg": 0}
    total_components_topk = {"pos": 0, "neg": 0}
    total_components_area_kept = {"pos": 0, "neg": 0}
    for source, output in jobs:
        reference = sitk.ReadImage(str(source))
        raw = sitk.GetArrayFromImage(reference) > 0
        spacing = reference.GetSpacing()
        processed, components_before, components_topk, components_area_kept = (
            erode_and_keep_topk(
            raw,
            args.erosion_radius,
            args.top_k,
            args.connectivity,
            float(spacing[0] * spacing[1]),
            args.min_area_mm2,
            )
        )
        write_like(processed, reference, output)

        prompt_type = "pos" if source.name.startswith("pos_") else "neg"
        before = int(raw.sum())
        after = int(processed.sum())
        total_before[prompt_type] += before
        total_after[prompt_type] += after
        total_components_before[prompt_type] += components_before
        total_components_topk[prompt_type] += components_topk
        total_components_area_kept[prompt_type] += components_area_kept
        print(
            f"test/{source.parent.name}/{prompt_type}: "
            f"voxels={before}->{after}, "
            f"components={components_before}->{components_topk}"
            f"->{components_area_kept} (all->top{args.top_k}->area-filtered)"
        )

    print(
        f"Done: {len(patients)} test patients; radius={args.erosion_radius}, "
        f"top_k={args.top_k}, min_area={args.min_area_mm2:g} mm2, "
        f"connectivity={args.connectivity}; "
        f"pos={total_before['pos']}->{total_after['pos']}, "
        f"components={total_components_before['pos']}->"
        f"{total_components_topk['pos']}->"
        f"{total_components_area_kept['pos']}; "
        f"neg={total_before['neg']}->{total_after['neg']}, "
        f"components={total_components_before['neg']}->"
        f"{total_components_topk['neg']}->"
        f"{total_components_area_kept['neg']}; "
        f"outputs=pos_erode{args.erosion_radius}_top{args.top_k}"
        f"_min{area_tag}mm2.nii.gz/neg_erode{args.erosion_radius}"
        f"_top{args.top_k}_min{area_tag}mm2.nii.gz"
    )


if __name__ == "__main__":
    main()
