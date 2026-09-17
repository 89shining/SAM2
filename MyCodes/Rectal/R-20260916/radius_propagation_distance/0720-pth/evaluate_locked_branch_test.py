#!/usr/bin/env python3
"""Evaluate each validation-locked branch/radius configuration on test only.

The script never selects a gate or radius.  It consumes the validation-locked
``per_radius_best`` table, writes case-level Dice/HD95 plus means, and saves one
header-preserving CTV_pred.nii.gz per test patient for every locked radius.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def patient_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse patient identifier: {path}")
    return int(match.group(1))


def read_mask(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(bool), image


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum() + b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


def hd95(a: np.ndarray, b: np.ndarray, spacing_xyz: tuple[float, ...]) -> float:
    """Existing experiment formula; intentionally unchanged."""
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


def gate_value(label: str) -> float:
    return math.inf if label == "inf" else float(label)


def write_prediction(path: Path, array: np.ndarray, reference: sitk.Image) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("pos", "neg"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run_root.resolve()
    data = run / "Prompt_mask"
    locked_path = run / "metrics" / f"LOCKED_{args.kind.upper()}_VALIDATION_CONFIG.json"
    locked = json.loads(locked_path.read_text())
    per_radius = {int(row["train_radius_mm"]): str(row["selected_gate_mm"]) for row in locked["per_radius_best"]}
    test_cases = sorted((data / "test").glob("p_*"), key=patient_number)
    if len(test_cases) != 36:
        raise RuntimeError(f"Expected 36 test cases, found {len(test_cases)}")

    rows: list[dict[str, object]] = []
    for radius, gate_label in sorted(per_radius.items()):
        gate = gate_value(gate_label)
        cache = run / "probability_cache" / "test" / args.kind / f"r{radius:02d}mm" / f"test_r{radius:02d}mm"
        if not (cache.parent / "DONE").is_file():
            raise RuntimeError(f"Missing completed test cache for {args.kind} r={radius}")
        prompt_name = f"{args.kind}_prompt_disk{radius}mm.nii.gz"
        for case in test_cases:
            patient = case.name
            gt, reference = read_mask(case / "CTV.nii.gz")
            nnunet, _ = read_mask(case / "nnunet.nii.gz")
            prompt_all, _ = read_mask(case / prompt_name)
            item = np.load(cache / f"{patient}.npz")
            selected = item["selected_prompt_slices"].astype(int)
            prompt = np.zeros_like(prompt_all)
            prompt[selected] = prompt_all[selected]
            probability = item["probability"].astype(np.float32)
            if prompt.any():
                distance = ndimage.distance_transform_edt(
                    ~prompt, sampling=tuple(reversed(reference.GetSpacing()))
                )
            else:
                distance = None
            disk_only = (nnunet | prompt) if args.kind == "pos" else (nnunet & ~prompt)
            if distance is None:
                prediction = nnunet.copy()
            else:
                gate_mask = np.ones_like(prompt, dtype=bool) if math.isinf(gate) else distance <= gate
                if args.kind == "pos":
                    prediction = nnunet | prompt | ((probability >= 0.5) & gate_mask & ~nnunet & ~prompt)
                else:
                    prediction = nnunet & ~(prompt | ((probability < 0.5) & gate_mask & nnunet & ~prompt))
            if gate == 0 and not np.array_equal(prediction, disk_only):
                raise RuntimeError(f"m=0 is not Disk-only for {args.kind}/{patient}/r{radius}")
            output = run / "TestResults" / args.kind / f"r{radius:02d}mm_m{gate_label}" / patient / "CTV_pred.nii.gz"
            write_prediction(output, prediction, reference)
            rows.append({
                "kind": args.kind, "patient": patient, "train_radius_mm": radius,
                "locked_gate_mm": gate_label, "dice": dice(prediction, gt),
                "hd95_mm": hd95(prediction, gt, reference.GetSpacing()),
                "prompt_voxels": int(prompt.sum()),
            })

    metrics = run / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    per_case_path = metrics / f"test_{args.kind}_per_case.csv"
    with per_case_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    profile = []
    for radius, gate_label in sorted(per_radius.items()):
        group = [row for row in rows if row["train_radius_mm"] == radius]
        profile.append({
            "kind": args.kind, "train_radius_mm": radius, "locked_gate_mm": gate_label,
            "mean_dice": float(np.mean([float(row["dice"]) for row in group])),
            "mean_hd95_mm": float(np.mean([float(row["hd95_mm"]) for row in group])),
            "cases": len(group),
        })
    with (metrics / f"test_{args.kind}_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile[0]))
        writer.writeheader()
        writer.writerows(profile)
    (run / f"TEST_{args.kind.upper()}_EVALUATION_COMPLETE").touch()


if __name__ == "__main__":
    main()
