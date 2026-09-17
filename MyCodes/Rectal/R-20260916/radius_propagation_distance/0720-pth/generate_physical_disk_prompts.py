#!/usr/bin/env python3
"""Build isolated physical-disk prompts for the R-20260916 experiment.

Pipeline per axial slice and per error type:
raw -> erosion(radius=2 pixels) -> largest three 8-connected components
    -> component area >= 50 mm^2 -> dilation(radius=2 pixels)
    -> one physical-radius disk per postprocessed component.

Disks are deliberately not clipped to a component; only the image FOV clips.

The source tree is read-only. Base images are symlinked into the experiment tree;
all derived masks and metrics are written below ``--output-root``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def read_mask(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image) > 0, image


def write_mask(mask: np.ndarray, reference: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), True)


def pixel_disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def preprocess_slice(mask: np.ndarray, pixel_area_mm2: float) -> tuple[np.ndarray, dict[str, int]]:
    eroded = ndimage.binary_erosion(mask, structure=pixel_disk(2))
    labels, count = ndimage.label(eroded, structure=CONNECTIVITY_8)
    components = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        components.append((int(component.sum()), component))
    components.sort(key=lambda item: item[0], reverse=True)
    top3 = components[:3]
    kept = [component for pixels, component in top3 if pixels * pixel_area_mm2 >= 50.0]
    retained_eroded = np.zeros_like(mask, dtype=bool)
    for component in kept:
        retained_eroded |= component
    result = ndimage.binary_dilation(retained_eroded, structure=pixel_disk(2))
    return result, {
        "raw_pixels": int(mask.sum()),
        "eroded_pixels": int(eroded.sum()),
        "eroded_components": int(count),
        "top3_components": int(len(top3)),
        "area50_components": int(len(kept)),
        "filtered_pixels": int(result.sum()),
    }


def physical_disk_prompt_grid(
    mask: np.ndarray,
    spacing_xy: tuple[float, float],
    radii_mm: list[float] | tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Generate all radii after labeling and center selection only once."""
    sx, sy = spacing_xy
    radii = [float(radius) for radius in radii_mm]
    results = {radius: np.zeros_like(mask, dtype=bool) for radius in radii}
    for z in range(mask.shape[0]):
        labels, count = ndimage.label(mask[z], structure=CONNECTIVITY_8)
        for label_id in range(1, count + 1):
            component = labels == label_id
            distance = ndimage.distance_transform_edt(
                component, sampling=(sy, sx)
            )
            cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
            yy, xx = np.ogrid[: component.shape[0], : component.shape[1]]
            squared_distance = ((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2
            for radius in radii:
                results[radius][z] |= squared_distance <= radius**2
    return results


def uniform_slice_subset(mask: np.ndarray, fraction: float) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    frames = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
    if not len(frames):
        return result
    keep_count = max(1, int(np.ceil(len(frames) * fraction)))
    positions = np.rint(np.linspace(0, len(frames) - 1, keep_count)).astype(int)
    result[frames[np.unique(positions)]] = mask[frames[np.unique(positions)]]
    return result


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum() + b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


def hd95(a: np.ndarray, b: np.ndarray, spacing_xyz: tuple[float, ...]) -> float:
    if not a.any() or not b.any():
        return 0.0 if a.any() == b.any() else float("inf")
    union = a | b
    points = np.argwhere(union)
    lo = np.maximum(points.min(axis=0) - 1, 0)
    hi = np.minimum(points.max(axis=0) + 2, union.shape)
    crop = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    a, b = a[crop], b[crop]
    structure = ndimage.generate_binary_structure(3, 1)
    sa = a ^ ndimage.binary_erosion(a, structure=structure, border_value=0)
    sb = b ^ ndimage.binary_erosion(b, structure=structure, border_value=0)
    sampling = tuple(reversed(spacing_xyz))
    da = ndimage.distance_transform_edt(~sa, sampling=sampling)
    db = ndimage.distance_transform_edt(~sb, sampling=sampling)
    return float(np.percentile(np.concatenate((db[sa], da[sb])), 95))


def ensure_links(source_case: Path, target_case: Path) -> None:
    target_case.mkdir(parents=True, exist_ok=True)
    for name in ("image.nii.gz", "CTV.nii.gz", "nnunet.nii.gz", "pos_raw.nii.gz", "neg_raw.nii.gz"):
        link = target_case / name
        if not link.exists():
            link.symlink_to(source_case / name)


def radius_tag(radius: float) -> str:
    return f"{radius:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--write-combined-diagnostic", action="store_true",
                        help="Optional diagnostic only; never used for selection.")
    parser.add_argument("--radii-mm", type=float, nargs="+", default=(2, 3, 4, 5, 6, 8))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_data = args.source_root
    dst_data = args.output_root / "Prompt_mask"
    metrics = args.output_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split_json.read_text())
    validation = set(split["validation"])
    component_rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    for subset in ("train", "test"):
        cases = sorted((src_data / subset).glob("p_*"), key=lambda path: int(path.name.split("_")[1]))
        for source_case in cases:
            target_case = dst_data / subset / source_case.name
            ensure_links(source_case, target_case)
            generated: dict[str, dict[float, np.ndarray]] = {"pos": {}, "neg": {}}
            filtered_masks: dict[str, np.ndarray] = {}
            reference = sitk.ReadImage(str(source_case / "CTV.nii.gz"))
            sx, sy, _ = reference.GetSpacing()
            for kind in ("pos", "neg"):
                raw, raw_ref = read_mask(source_case / f"{kind}_raw.nii.gz")
                filtered = np.zeros_like(raw, dtype=bool)
                totals = {key: 0 for key in ("raw_pixels", "eroded_pixels", "eroded_components", "top3_components", "area50_components", "filtered_pixels")}
                retained_slices = 0
                for z in range(raw.shape[0]):
                    processed, stats = preprocess_slice(raw[z], sx * sy)
                    filtered[z] = processed
                    retained_slices += int(processed.any())
                    for key, value in stats.items():
                        totals[key] += value
                filtered_masks[kind] = filtered
                write_mask(filtered, raw_ref, target_case / f"{kind}_prompt_erode2_top3_area50.nii.gz")
                prompt_grid = physical_disk_prompt_grid(filtered, (sx, sy), args.radii_mm)
                for radius in args.radii_mm:
                    prompt = prompt_grid[float(radius)]
                    generated[kind][radius] = prompt
                    write_mask(prompt, raw_ref, target_case / f"{kind}_prompt_disk{radius_tag(radius)}mm.nii.gz")
                component_rows.append({
                    "subset": subset, "patient": source_case.name, "kind": kind,
                    "spacing_x_mm": sx, "spacing_y_mm": sy, "retained_slices": retained_slices,
                    **totals,
                })
            # The test set remains unscored until every choice is locked on validation.
            if (not args.write_combined_diagnostic or subset != "train"
                    or source_case.name not in validation):
                continue
            gt, _ = read_mask(source_case / "CTV.nii.gz")
            nnunet, _ = read_mask(source_case / "nnunet.nii.gz")
            for radius in args.radii_mm:
                for fraction in (0.25, 0.50, 0.75, 1.00):
                    pos = uniform_slice_subset(generated["pos"][radius], fraction)
                    neg = uniform_slice_subset(generated["neg"][radius], fraction)
                    prediction = (nnunet | pos) & ~neg
                    direct_rows.append({
                        "patient": source_case.name, "radius_mm": radius,
                        "prompt_fraction": fraction, "dice": dice(prediction, gt),
                        "hd95_mm": hd95(prediction, gt, reference.GetSpacing()),
                        "pos_prompt_slices": int(pos.reshape(pos.shape[0], -1).any(axis=1).sum()),
                        "neg_prompt_slices": int(neg.reshape(neg.shape[0], -1).any(axis=1).sum()),
                        "pos_prompt_voxels": int(pos.sum()), "neg_prompt_voxels": int(neg.sum()),
                    })
    outputs = [(metrics / "prompt_preprocessing_stats.csv", component_rows)]
    if args.write_combined_diagnostic:
        outputs.append((metrics / "combined_direct_validation_DIAGNOSTIC_ONLY.csv", direct_rows))
    for path, rows in outputs:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = []
    if args.write_combined_diagnostic:
        for radius in args.radii_mm:
            for fraction in (0.25, 0.50, 0.75, 1.00):
                group = [row for row in direct_rows if row["radius_mm"] == radius and row["prompt_fraction"] == fraction]
                summary.append({
                    "radius_mm": radius, "prompt_fraction": fraction,
                    "mean_dice": float(np.mean([row["dice"] for row in group])),
                    "mean_hd95_mm": float(np.mean([row["hd95_mm"] for row in group])),
                    "patients": len(group),
                })
        (metrics / "combined_direct_validation_DIAGNOSTIC_ONLY.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
