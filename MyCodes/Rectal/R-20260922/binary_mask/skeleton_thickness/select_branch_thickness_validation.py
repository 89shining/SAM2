#!/usr/bin/env python3
"""Select m*(thickness) and thickness* for one branch using validation data only.

This script is deliberately downstream of training: it accepts one fixed
``best.pth`` probability cache per training thickness and never inspects epochs.
It therefore implements the locked order epoch -> m*(r) -> r*.
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


THICKNESSES = (0, 2, 4, 6, 8, 10, 12)
GATES: tuple[float, ...] = (0, 5, 10, 15, 20, 30, 40, 60, math.inf)
EPS = 1e-12


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
    """Existing experiment formula; intentionally kept unchanged."""
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


def gate_label(value: float) -> str:
    return "inf" if math.isinf(value) else str(int(value))


def pick(rows: list[dict], thickness_key: str, gate_key: str = "gate_mm") -> dict:
    """Dice-equivalence (0.001), then HD95, then smaller named thickness/gate."""
    dmax = max(float(row["mean_dice"]) for row in rows)
    equivalent = [row for row in rows if float(row["mean_dice"]) >= dmax - 0.001]
    best_hd = min(float(row["mean_hd95_mm"]) for row in equivalent)
    hd_best = [row for row in equivalent if abs(float(row["mean_hd95_mm"]) - best_hd) <= EPS]
    return min(hd_best, key=lambda row: (float(row[thickness_key]), float(row[gate_key])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("pos", "neg"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--thicknesses-mm", type=float, nargs="+", default=THICKNESSES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = json.loads(args.split_json.read_text())["validation"]
    validation = sorted(validation, key=lambda name: patient_number(Path(name)))
    args.metrics_root.mkdir(parents=True, exist_ok=True)
    per_case: list[dict] = []

    for thickness in args.thicknesses_mm:
        cache_dir = args.cache_root / args.kind / f"t{int(thickness):02d}mm"
        for patient in validation:
            case = args.data_root / "train" / patient
            gt, reference = read_mask(case / "CTV.nii.gz")
            nnunet, _ = read_mask(case / "nnunet.nii.gz")
            item = np.load(cache_dir / f"{patient}.npz")
            selected = item["selected_prompt_slices"].astype(int)
            prompt = item["prompt_mask"].astype(bool)
            probability = item["probability"].astype(np.float32)
            if prompt.any():
                distance = ndimage.distance_transform_edt(
                    ~prompt, sampling=tuple(reversed(reference.GetSpacing()))
                )
            else:
                distance = None
            disk_only = (nnunet | prompt) if args.kind == "pos" else (nnunet & ~prompt)
            for gate in GATES:
                # An empty prompt is a whole-volume no-op, including at infinity.
                if distance is None:
                    prediction = nnunet.copy()
                elif math.isinf(gate):
                    gate_mask = np.ones_like(prompt, dtype=bool)
                    if args.kind == "pos":
                        prediction = nnunet | prompt | ((probability >= 0.5) & gate_mask & ~nnunet & ~prompt)
                    else:
                        prediction = nnunet & ~(prompt | ((probability < 0.5) & gate_mask & nnunet & ~prompt))
                else:
                    gate_mask = distance <= gate
                    if args.kind == "pos":
                        prediction = nnunet | prompt | ((probability >= 0.5) & gate_mask & ~nnunet & ~prompt)
                    else:
                        prediction = nnunet & ~(prompt | ((probability < 0.5) & gate_mask & nnunet & ~prompt))
                if gate == 0 and not np.array_equal(prediction, disk_only):
                    raise RuntimeError(f"m=0 is not Disk-only for {args.kind}/{patient}/t{thickness}")
                per_case.append({
                    "kind": args.kind, "patient": patient, "thickness_mm": thickness,
                    "gate_mm": gate_label(gate), "dice": dice(prediction, gt),
                    "hd95_mm": hd95(prediction, gt, reference.GetSpacing()),
                    "prompt_voxels": int(prompt.sum()),
                })

    per_case_path = args.metrics_root / f"validation_{args.kind}_per_case.csv"
    with per_case_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case[0]))
        writer.writeheader(); writer.writerows(per_case)
    summary: list[dict] = []
    for thickness in args.thicknesses_mm:
        for gate in GATES:
            group = [row for row in per_case if row["thickness_mm"] == thickness and row["gate_mm"] == gate_label(gate)]
            summary.append({"kind": args.kind, "thickness_mm": thickness, "gate_mm": gate_label(gate),
                            "mean_dice": float(np.mean([row["dice"] for row in group])),
                            "mean_hd95_mm": float(np.mean([row["hd95_mm"] for row in group])), "cases": len(group)})
    with (args.metrics_root / f"validation_{args.kind}_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    profile = [pick([row for row in summary if row["thickness_mm"] == thickness], "gate_mm") for thickness in args.thicknesses_mm]
    for row in profile:
        row["selected_gate_mm"] = row.pop("gate_mm")
    with (args.metrics_root / f"validation_{args.kind}_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile[0])); writer.writeheader(); writer.writerows(profile)
    locked = pick(profile, "thickness_mm", "selected_gate_mm")
    (args.metrics_root / f"LOCKED_{args.kind.upper()}_VALIDATION_CONFIG.json").write_text(json.dumps({
        "selection_order": "fixed best.pth -> m*(r) -> r*", "dice_equivalence": 0.001,
        "tie_break_order": ["lowest mean HD95", "smaller mm"],
        "locked": locked, "per_radius_best": profile,
    }, indent=2))
    print(json.dumps({"locked": locked, "per_radius_best": profile}, indent=2))


if __name__ == "__main__":
    main()
