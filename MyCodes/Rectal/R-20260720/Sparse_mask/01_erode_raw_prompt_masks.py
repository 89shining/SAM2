#!/usr/bin/env python3
"""Postprocess train and test raw prompts slice by slice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


DEFAULT_PROMPT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)
DATA_SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply axial-slice 2D erosion to train and test raw prompts, "
            "then retain the largest components, filter them by physical "
            "area, and finally apply axial-slice 2D dilation."
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
        "--dilation-radius",
        type=int,
        default=0,
        help=(
            "Final in-plane dilation radius in pixels; "
            "0 keeps the area-filtered mask unchanged."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs with the same radius.",
    )
    return parser.parse_args()


def erode_and_keep_topk(
    mask_zyx: np.ndarray,
    erosion_radius: int,
    top_k: int,
    connectivity: int,
    pixel_area_mm2: float,
    min_area_mm2: float,
    dilation_radius: int,
) -> tuple[np.ndarray, int, int, int]:
    """Erode, retain Top-K, area-filter, then dilate on every slice."""
    mask = mask_zyx > 0
    if erosion_radius > 0:
        yy, xx = np.ogrid[
            -erosion_radius : erosion_radius + 1,
            -erosion_radius : erosion_radius + 1,
        ]
        erosion_disk = (xx * xx + yy * yy) <= erosion_radius * erosion_radius
    else:
        erosion_disk = None
    if dilation_radius > 0:
        yy, xx = np.ogrid[
            -dilation_radius : dilation_radius + 1,
            -dilation_radius : dilation_radius + 1,
        ]
        dilation_disk = (xx * xx + yy * yy) <= dilation_radius * dilation_radius
    else:
        dilation_disk = None
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
        if erosion_disk is not None:
            current = ndimage.binary_erosion(
                current,
                structure=erosion_disk,
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
        filtered = np.isin(labels, qualifying_labels)
        if dilation_disk is not None:
            filtered = ndimage.binary_dilation(
                filtered,
                structure=dilation_disk,
                border_value=0,
            )
        output[z] = filtered
        components_area_kept += len(qualifying_labels)

    return output, components_before, components_topk, components_area_kept


def write_like(mask_zyx: np.ndarray, reference: sitk.Image, output: Path) -> None:
    image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(output), useCompression=True)


def get_patients(prompt_root: Path) -> list[tuple[str, Path]]:
    """Return all p_* patient folders from both train and test."""
    patients: list[tuple[str, Path]] = []
    for split in DATA_SPLITS:
        split_dir = prompt_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split.capitalize()} directory does not exist: {split_dir}"
            )
        split_patients = sorted(p for p in split_dir.glob("p_*") if p.is_dir())
        if not split_patients:
            raise RuntimeError(f"No p_* patient folders found in: {split_dir}")
        patients.extend((split, patient) for patient in split_patients)
    return patients


def main() -> None:
    args = parse_args()
    if args.erosion_radius < 0:
        raise ValueError("--erosion-radius must be >= 0")
    if args.dilation_radius < 0:
        raise ValueError("--dilation-radius must be >= 0")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.min_area_mm2 < 0:
        raise ValueError("--min-area-mm2 must be >= 0")

    patients = get_patients(args.prompt_root)
    patient_counts = {
        split: sum(patient_split == split for patient_split, _ in patients)
        for split in DATA_SPLITS
    }
    area_tag = f"{args.min_area_mm2:g}".replace(".", "p")
    jobs: list[tuple[str, Path, Path]] = []
    for split, patient in patients:
        for prompt_type in ("pos", "neg"):
            source = patient / f"{prompt_type}_raw.nii.gz"
            output = (
                patient
                / (
                    f"{prompt_type}_erode{args.erosion_radius}_top{args.top_k}"
                    f"_min{area_tag}mm2_dilate{args.dilation_radius}.nii.gz"
                )
            )
            if not source.is_file():
                raise FileNotFoundError(f"Missing raw prompt: {source}")
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {output}")
            jobs.append((split, source, output))

    total_before = {"pos": 0, "neg": 0}
    total_after = {"pos": 0, "neg": 0}
    total_components_before = {"pos": 0, "neg": 0}
    total_components_topk = {"pos": 0, "neg": 0}
    total_components_area_kept = {"pos": 0, "neg": 0}
    for split, source, output in jobs:
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
                args.dilation_radius,
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
            f"{split}/{source.parent.name}/{prompt_type}: "
            f"voxels={before}->{after}, "
            f"components={components_before}->{components_topk}"
            f"->{components_area_kept} (all->top{args.top_k}->area-filtered)"
        )

    print(
        f"Done: {len(patients)} patients "
        f"(train={patient_counts['train']}, test={patient_counts['test']}); "
        f"radius={args.erosion_radius}, "
        f"top_k={args.top_k}, min_area={args.min_area_mm2:g} mm2, "
        f"dilation_radius={args.dilation_radius}, "
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
        f"_min{area_tag}mm2_dilate{args.dilation_radius}.nii.gz/"
        f"neg_erode{args.erosion_radius}_top{args.top_k}"
        f"_min{area_tag}mm2_dilate{args.dilation_radius}.nii.gz"
    )


if __name__ == "__main__":
    main()
