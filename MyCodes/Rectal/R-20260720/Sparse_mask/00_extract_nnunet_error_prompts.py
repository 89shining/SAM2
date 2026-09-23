#!/usr/bin/env python3
"""Build train/test SAM2 prompt-mask data from nnU-Net segmentation errors.

For each patient:
  pos_raw = GT foreground missed by nnU-Net
  neg_raw = nnU-Net foreground outside GT
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


RESULTS_ROOT = Path(
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset013_RectalCTV146pCrop/nnUNetTrainer__nnUNetPlans__3d_fullres"
)
RAW_ROOT = Path(
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/"
    "Dataset013_RectalCTV146pCrop"
)
OUTPUT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)


@dataclass(frozen=True)
class Case:
    split: str
    patient_id: int
    image: Path
    gt: Path
    prediction: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/test nnU-Net error prompt masks.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing patient files.")
    return parser.parse_args()


def is_nifti(path: Path) -> bool:
    return path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))


def case_key(filename: str) -> str:
    """Return nnU-Net case name, removing NIfTI suffix and optional channel suffix."""
    name = filename
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return re.sub(r"_\d{4}$", "", name)


def patient_number(case_name: str) -> int:
    matches = re.findall(r"\d+", case_name)
    if not matches:
        raise ValueError(f"Cannot extract patient number from case name: {case_name}")
    return int(matches[-1])


def index_by_case(paths: list[Path], description: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in sorted(paths):
        key = case_key(path.name)
        if key in indexed:
            raise ValueError(f"Duplicate {description} for {key}: {indexed[key]} and {path}")
        indexed[key] = path
    return indexed


def files_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    return [p for p in folder.iterdir() if is_nifti(p)]


def collect_train_predictions(results_root: Path) -> list[Path]:
    predictions: list[Path] = []
    for fold in range(5):
        validation_dir = results_root / f"fold_{fold}" / "validation"
        predictions.extend(files_in(validation_dir))
    if not predictions:
        raise RuntimeError("No training validation predictions found in folds 0-4.")
    return predictions


def collect_test_predictions(results_root: Path) -> list[Path]:
    test_dir = results_root / "Revise_testresults_5folds"
    predictions = files_in(test_dir)
    if not predictions:
        raise RuntimeError(f"No test predictions found in: {test_dir}")
    return predictions


def build_cases(
    split: str, predictions: list[Path], image_dir: Path, gt_dir: Path
) -> list[Case]:
    pred_index = index_by_case(predictions, f"{split} prediction")
    gt_index = index_by_case(files_in(gt_dir), f"{split} GT")

    # nnU-Net images may be named CTV_001_0000.nii.gz; this maps them to CTV_001.
    image_paths = files_in(image_dir)
    image_index = index_by_case(image_paths, f"{split} image")

    missing_gt = sorted(set(pred_index) - set(gt_index))
    missing_image = sorted(set(pred_index) - set(image_index))
    if missing_gt or missing_image:
        details = []
        if missing_gt:
            details.append(f"missing GT ({len(missing_gt)}): {missing_gt[:20]}")
        if missing_image:
            details.append(f"missing image ({len(missing_image)}): {missing_image[:20]}")
        raise FileNotFoundError("; ".join(details))

    ids: dict[int, str] = {}
    cases: list[Case] = []
    for key, prediction in sorted(pred_index.items()):
        pid = patient_number(key)
        if pid in ids:
            raise ValueError(f"Duplicate patient ID {pid} in {split}: {ids[pid]} and {key}")
        ids[pid] = key
        cases.append(Case(split, pid, image_index[key], gt_index[key], prediction))
    return cases


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def validate_geometry(cases: list[Case]) -> None:
    for case in cases:
        gt = sitk.ReadImage(str(case.gt))
        prediction = sitk.ReadImage(str(case.prediction))
        image = sitk.ReadImage(str(case.image))
        if not same_geometry(gt, prediction):
            raise ValueError(f"GT/prediction geometry mismatch: {case.prediction.name}")
        if not same_geometry(gt, image):
            raise ValueError(f"GT/image geometry mismatch: {case.image.name}")


def write_mask(mask_zyx: np.ndarray, reference: sitk.Image, path: Path) -> None:
    image = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), useCompression=True)


def process_case(case: Case, output_root: Path, overwrite: bool) -> tuple[int, int]:
    patient_dir = output_root / case.split / f"p_{case.patient_id}"
    outputs = {
        "image": patient_dir / "image.nii.gz",
        "gt": patient_dir / "CTV.nii.gz",
        "prediction": patient_dir / "nnunet.nii.gz",
        "pos": patient_dir / "pos_raw.nii.gz",
        "neg": patient_dir / "neg_raw.nii.gz",
    }
    existing = [p for p in outputs.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output exists for {case.split}/p_{case.patient_id}; use --overwrite: {existing[0]}"
        )

    patient_dir.mkdir(parents=True, exist_ok=True)
    # Copy original inputs first, retaining their bytes and metadata.
    shutil.copy2(case.image, outputs["image"])
    shutil.copy2(case.gt, outputs["gt"])
    shutil.copy2(case.prediction, outputs["prediction"])

    gt_image = sitk.ReadImage(str(outputs["gt"]))
    prediction_image = sitk.ReadImage(str(outputs["prediction"]))
    gt = sitk.GetArrayFromImage(gt_image) > 0
    prediction = sitk.GetArrayFromImage(prediction_image) > 0
    pos = gt & ~prediction
    neg = prediction & ~gt
    write_mask(pos, gt_image, outputs["pos"])
    write_mask(neg, gt_image, outputs["neg"])
    return int(pos.sum()), int(neg.sum())


def main() -> None:
    args = parse_args()
    train_cases = build_cases(
        "train",
        collect_train_predictions(args.results_root),
        args.raw_root / "imagesTr",
        args.raw_root / "labelsTr",
    )
    test_cases = build_cases(
        "test",
        collect_test_predictions(args.results_root),
        args.raw_root / "imagesTs",
        args.raw_root / "labelsTs",
    )
    all_cases = train_cases + test_cases

    # Check all input pairs before writing anything to prevent a partial dataset.
    validate_geometry(all_cases)
    print(f"Validated {len(train_cases)} train and {len(test_cases)} test cases.")

    totals = {"train": [0, 0], "test": [0, 0]}
    for case in all_cases:
        pos_count, neg_count = process_case(case, args.output_root, args.overwrite)
        totals[case.split][0] += pos_count
        totals[case.split][1] += neg_count
        print(
            f"{case.split}/p_{case.patient_id}: "
            f"pos_raw={pos_count}, neg_raw={neg_count}"
        )

    print(
        f"Done. train: {len(train_cases)} cases, pos={totals['train'][0]}, "
        f"neg={totals['train'][1]}; test: {len(test_cases)} cases, "
        f"pos={totals['test'][0]}, neg={totals['test'][1]}\nOutput: {args.output_root}"
    )


if __name__ == "__main__":
    main()
