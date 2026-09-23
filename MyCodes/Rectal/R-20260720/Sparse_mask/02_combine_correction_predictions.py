#!/usr/bin/env python3
"""Combine positive/negative correction predictions under nnU-Net constraints.

Supported ablation modes:
    both:     (pos_predict AND NOT nnunet) OR (neg_predict AND nnunet)
    pos:      nnunet OR (pos_predict AND NOT nnunet)
    neg:      neg_predict AND nnunet
    none:     (pos_prompt AND NOT nnunet) OR (nnunet AND NOT neg_prompt)
    none_pos: nnunet OR (pos_prompt AND NOT nnunet)
    none_neg: nnunet AND NOT neg_prompt
    baseline: nnunet

This script performs no connected-component or morphology processing.  Its output
is intended to be consumed by ``03_postprocess_final_prediction.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DEFAULT_PATIENT_ROOT = Path(
    "D:\A-project\Rectal\CTV\146p\Revise_SAM2\TestData"
)

DEFAULT_OUTPUT_NAMES = {
    "both": "final_predict_raw.nii.gz",
    "pos": "pos_only_predict_raw.nii.gz",
    "neg": "neg_only_predict_raw.nii.gz",
    "none": "final_predict_none_raw.nii.gz",
    "none_pos": "final_predict_none_pos_raw.nii.gz",
    "none_neg": "final_predict_none_neg_raw.nii.gz",
    "baseline": "nnunet_baseline_raw.nii.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine pos/neg predictions using the nnU-Net mask as a domain constraint."
    )
    parser.add_argument("--patient-root", type=Path, default=DEFAULT_PATIENT_ROOT)
    parser.add_argument("--nnunet-name", default="nnunet.nii.gz")
    parser.add_argument("--pos-name", default="pos_predict.nii.gz")
    parser.add_argument("--neg-name", default="neg_predict.nii.gz")
    parser.add_argument(
        "--mode",
        choices=("both", "pos", "neg", "none", "none_pos", "none_neg", "baseline"),
        default="both",
        help="Ablation mode (default: both).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename. If omitted, a mode-specific filename is used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def patient_dirs(root: Path) -> list[Path]:
    """Accept either a folder containing p_* or a root containing train/test/p_*."""
    if not root.is_dir():
        raise FileNotFoundError(f"Patient root does not exist: {root}")
    direct = sorted(path for path in root.glob("p_*") if path.is_dir())
    if direct:
        return direct
    nested = sorted(path for path in root.glob("*/p_*") if path.is_dir())
    if not nested:
        raise RuntimeError(f"No p_* patient folders found under: {root}")
    return nested


def read_binary(path: Path) -> tuple[sitk.Image, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image) > 0


def write_like(mask_zyx: np.ndarray, reference: sitk.Image, path: Path) -> None:
    output = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(path), useCompression=True)


def main() -> None:
    args = parse_args()
    patients = patient_dirs(args.patient_root)
    output_name = args.output_name or DEFAULT_OUTPUT_NAMES[args.mode]

    jobs: list[tuple[Path, Path]] = []
    for patient in patients:
        output = patient / output_name
        required_names = [args.nnunet_name]
        if args.mode in ("both", "pos", "none", "none_pos"):
            required_names.append(args.pos_name)
        if args.mode in ("both", "neg", "none", "none_neg"):
            required_names.append(args.neg_name)
        for name in required_names:
            if not (patient / name).is_file():
                raise FileNotFoundError(f"Missing input: {patient / name}")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        jobs.append((patient, output))

    for patient, output in jobs:
        nn_image, nnunet = read_binary(patient / args.nnunet_name)
        pos = None
        neg = None
        if args.mode in ("both", "pos", "none", "none_pos"):
            pos_image, pos = read_binary(patient / args.pos_name)
            if not same_geometry(nn_image, pos_image):
                raise ValueError(f"nnU-Net/pos geometry mismatch: {patient}")
        if args.mode in ("both", "neg", "none", "none_neg"):
            neg_image, neg = read_binary(patient / args.neg_name)
            if not same_geometry(nn_image, neg_image):
                raise ValueError(f"nnU-Net/neg geometry mismatch: {patient}")

        additions = (pos & ~nnunet) if pos is not None else np.zeros_like(nnunet)
        retained = (neg & nnunet) if neg is not None else nnunet.copy()
        if args.mode == "both":
            raw_final = additions | retained
        elif args.mode == "pos":
            raw_final = nnunet | additions
        elif args.mode == "neg":
            raw_final = retained
        elif args.mode == "none":
            # In no-network ablation, pos/neg inputs are correction masks from
            # preprocessing step 01, not full corrected segmentation outputs.
            removals = neg & nnunet
            retained = nnunet & ~removals
            raw_final = additions | retained
        elif args.mode == "none_pos":
            raw_final = nnunet | additions
        elif args.mode == "none_neg":
            removals = neg & nnunet
            retained = nnunet & ~removals
            raw_final = retained
        else:
            raw_final = nnunet.copy()
        write_like(raw_final, nn_image, output)
        print(
            f"{patient.name}: mode={args.mode}, nnunet={int(nnunet.sum())}, "
            f"additions={int(additions.sum())}, retained={int(retained.sum())}, "
            f"raw_final={int(raw_final.sum())} -> {output.name}"
        )

    print(
        f"Done: mode={args.mode}, combined {len(jobs)} patients under "
        f"{args.patient_root}; output={output_name}"
    )


if __name__ == "__main__":
    main()
