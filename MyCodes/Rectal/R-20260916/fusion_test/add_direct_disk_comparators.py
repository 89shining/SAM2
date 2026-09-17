#!/usr/bin/env python3
"""Add POS-only, NEG-only, disk-direct, and fusion summaries without rerunning SAM2."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from run_fusion_prompt_fraction_test import (
    FRACTIONS,
    OUTPUT_ROOT,
    SCHEMES,
    branch_prediction,
    cache_dir,
    dice,
    fraction_label,
    hd95,
    write_mask,
)


def selected_disk(scheme: str, fraction: float, branch: str, case: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Return only the deterministic selected disk slices for one branch."""
    if fraction == 0:
        return np.zeros(shape, dtype=bool)
    radius = SCHEMES[scheme][branch]["r"]
    disk = sitk.GetArrayFromImage(
        sitk.ReadImage(str(case / f"{branch}_prompt_disk{radius}mm.nii.gz"))
    ).astype(bool)
    saved = np.load(cache_dir(scheme, fraction, branch) / f"test_r{radius:02d}mm" / f"{case.name}.npz")
    selected = saved["selected_prompt_slices"].astype(int)
    prompt = np.zeros_like(disk)
    prompt[selected] = disk[selected]
    return prompt


def main() -> None:
    all_summary: list[dict[str, object]] = []
    for scheme, spec in SCHEMES.items():
        cases = sorted((spec["root"] / "Prompt_mask" / "test").glob("p_*"), key=lambda p: int(p.name.split("_")[-1]))
        if len(cases) != 36:
            raise RuntimeError(f"{scheme}: expected 36 cases, found {len(cases)}")
        for fraction in FRACTIONS:
            folder = OUTPUT_ROOT / scheme / fraction_label(fraction)
            rows: list[dict[str, object]] = []
            totals = {method: {"dice": [], "hd95": []} for method in ("pos_only", "neg_only", "direct_disk", "fusion")}
            for case in cases:
                reference = sitk.ReadImage(str(case / "image.nii.gz"))
                nnunet = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "nnunet.nii.gz"))).astype(bool)
                gt = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "CTV.nii.gz"))).astype(bool)
                # Existing NIfTIs are exactly the SAM2 POS-only and NEG-only outputs.
                patient_dir = folder / case.name
                pos_only = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / "POS_pred.nii.gz"))).astype(bool)
                neg_only = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / "NEG_pred.nii.gz"))).astype(bool)
                fusion = sitk.GetArrayFromImage(sitk.ReadImage(str(patient_dir / "CTV_pred.nii.gz"))).astype(bool)
                pos_disk = selected_disk(scheme, fraction, "pos", case, nnunet.shape)
                neg_disk = selected_disk(scheme, fraction, "neg", case, nnunet.shape)
                # No SAM2 probability or m gate: POS disks add; NEG disks delete original nnU-Net voxels only.
                direct = (nnunet | pos_disk) & ~(nnunet & neg_disk)
                write_mask(patient_dir / "Direct_disk_pred.nii.gz", direct, reference)
                predictions = {"pos_only": pos_only, "neg_only": neg_only, "direct_disk": direct, "fusion": fusion}
                row: dict[str, object] = {"scheme": scheme, "prompt_fraction": fraction, "patient": case.name}
                for method, prediction in predictions.items():
                    method_dice = dice(prediction, gt)
                    method_hd95 = hd95(prediction, gt, reference.GetSpacing())
                    row[f"{method}_dice"] = method_dice
                    row[f"{method}_hd95_mm"] = method_hd95
                    totals[method]["dice"].append(method_dice)
                    totals[method]["hd95"].append(method_hd95)
                rows.append(row)
            with (folder / "comparison_per_case.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            for method, values in totals.items():
                all_summary.append({
                    "scheme": scheme, "prompt_fraction": fraction, "method": method,
                    "cases": len(cases), "mean_dice": float(np.mean(values["dice"])),
                    "mean_hd95_mm": float(np.mean(values["hd95"])),
                })
    with (OUTPUT_ROOT / "method_prompt_fraction_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary[0]))
        writer.writeheader()
        writer.writerows(all_summary)
    (OUTPUT_ROOT / "COMPARATOR_EVALUATION_COMPLETE").touch()


if __name__ == "__main__":
    main()
