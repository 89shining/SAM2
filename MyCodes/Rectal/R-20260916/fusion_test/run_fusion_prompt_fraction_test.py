#!/usr/bin/env python3
"""Formal test-only POS/NEG fusion across deterministic prompt fractions.

Validation has already locked (r, m) for each initialization scheme.  This
script never changes those values.  It evaluates 0/25/50/75/100% uniformly and
deterministically selected prompt slices, saves POS/NEG/fused NIfTI predictions,
and reports only Dice and HD95.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


RESULT_ROOT = Path("/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260916")
EXPERIMENT_ROOT = RESULT_ROOT / "radius_propagation_distance"
SOURCE_CODE_ROOT = Path("/home/wusi/SAM2/MyTrain/MyCodes/Rectal/R-20260916/radius_propagation_distance")
OUTPUT_ROOT = RESULT_ROOT / "fusion_test"
PYTHON = "/home/wusi/miniconda3/envs/sam2/bin/python"
GPU_IDS = (2, 3, 4, 5)  # GPU 6 is deliberately reserved.
MIN_FREE_MIB = 16_000
FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)

SCHEMES = {
    "0720-pth": {
        "root": EXPERIMENT_ROOT / "0720-pth",
        "code": SOURCE_CODE_ROOT / "0720-pth",
        "pos": {"r": 10, "m": 20},
        "neg": {"r": 10, "m": 20},
    },
    "small.pth": {
        "root": EXPERIMENT_ROOT / "small.pth",
        "code": SOURCE_CODE_ROOT / "small.pth",
        "pos": {"r": 2, "m": 30},
        "neg": {"r": 10, "m": 15},
    },
}


def fraction_label(fraction: float) -> str:
    return f"prompt_{int(round(fraction * 100)):02d}"


def gpu_state() -> list[tuple[int, int]]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [tuple(map(int, line.split(","))) for line in text.strip().splitlines()]


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


def write_mask(path: Path, mask: np.ndarray, reference: sitk.Image) -> None:
    image = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
    image.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def inference_tasks() -> list[tuple[str, float, str]]:
    return [
        (scheme, fraction, branch)
        for scheme in SCHEMES
        for fraction in FRACTIONS
        if fraction > 0
        for branch in ("pos", "neg")
    ]


def cache_dir(scheme: str, fraction: float, branch: str) -> Path:
    radius = SCHEMES[scheme][branch]["r"]
    return OUTPUT_ROOT / scheme / "probability_cache" / fraction_label(fraction) / branch / f"r{radius:02d}mm"


def run_inference() -> None:
    pending = [task for task in inference_tasks() if not (cache_dir(*task) / "DONE").is_file()]
    active: dict[int, tuple[subprocess.Popen, object, tuple[str, float, str]]] = {}
    logs = OUTPUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu, (process, handle, task) in list(active.items()):
            if process.poll() is None:
                continue
            handle.close()
            del active[gpu]
            if process.returncode:
                raise RuntimeError(f"Inference failed for {task} on GPU {gpu}")
            (cache_dir(*task) / "DONE").touch()
            print(f"[{time.strftime('%F %T')}] DONE {task} on GPU {gpu}", flush=True)
        ready = [(gpu, free) for gpu, free in gpu_state() if gpu in GPU_IDS and gpu not in active and free >= MIN_FREE_MIB]
        for gpu, free in sorted(ready, key=lambda item: item[1], reverse=True)[: len(GPU_IDS) - len(active)]:
            if not pending:
                break
            scheme, fraction, branch = pending.pop(0)
            spec = SCHEMES[scheme]
            radius = spec[branch]["r"]
            checkpoint_root = spec["root"] / "checkpoints" / branch / f"r{radius:02d}mm"
            output = cache_dir(scheme, fraction, branch)
            output.mkdir(parents=True, exist_ok=True)
            command = [
                PYTHON, str(spec["code"] / "infer_radius_grid.py"), "--kind", branch,
                "--data-root", str(spec["root"] / "Prompt_mask"),
                "--split-json", str(checkpoint_root / "split.json"), "--subset", "test",
                "--checkpoint", str(checkpoint_root / "checkpoints" / "best.pth"),
                "--output-root", str(output), "--test-radii", str(radius),
                "--input-size", "512", "--prompt-fraction", str(fraction), "--device", "cuda",
            ]
            handle = (logs / f"infer_{scheme}_{fraction_label(fraction)}_{branch}.log").open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["SAM2_PROJECT_ROOT"] = "/home/wusi/SAM2"
            active[gpu] = (subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle, (scheme, fraction, branch))
            print(f"[{time.strftime('%F %T')}] START {(scheme, fraction, branch)} on GPU {gpu} (free={free} MiB)", flush=True)
        if pending or active:
            time.sleep(15)
    (OUTPUT_ROOT / "INFERENCE_COMPLETE").touch()


def branch_prediction(scheme: str, fraction: float, branch: str, case: Path, nnunet: np.ndarray, reference: sitk.Image) -> np.ndarray:
    if fraction == 0:
        return nnunet.copy()
    radius = SCHEMES[scheme][branch]["r"]
    gate = float(SCHEMES[scheme][branch]["m"])
    prompt_all = sitk.GetArrayFromImage(sitk.ReadImage(str(case / f"{branch}_prompt_disk{radius}mm.nii.gz"))).astype(bool)
    item = np.load(cache_dir(scheme, fraction, branch) / f"test_r{radius:02d}mm" / f"{case.name}.npz")
    selected = item["selected_prompt_slices"].astype(int)
    prompt = np.zeros_like(prompt_all)
    prompt[selected] = prompt_all[selected]
    probability = item["probability"].astype(np.float32)
    if not prompt.any():
        return nnunet.copy()
    distance = ndimage.distance_transform_edt(~prompt, sampling=tuple(reversed(reference.GetSpacing())))
    gate_mask = distance <= gate
    if branch == "pos":
        return nnunet | prompt | ((probability >= 0.5) & gate_mask & ~nnunet & ~prompt)
    return nnunet & ~(prompt | ((probability < 0.5) & gate_mask & nnunet & ~prompt))


def evaluate() -> None:
    summaries: list[dict[str, object]] = []
    for scheme, spec in SCHEMES.items():
        cases = sorted((spec["root"] / "Prompt_mask" / "test").glob("p_*"), key=lambda path: int(path.name.split("_")[-1]))
        if len(cases) != 36:
            raise RuntimeError(f"{scheme}: expected 36 test cases, found {len(cases)}")
        for fraction in FRACTIONS:
            rows: list[dict[str, object]] = []
            folder = OUTPUT_ROOT / scheme / fraction_label(fraction)
            for case in cases:
                reference = sitk.ReadImage(str(case / "image.nii.gz"))
                nnunet = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "nnunet.nii.gz"))).astype(bool)
                gt = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "CTV.nii.gz"))).astype(bool)
                pos = branch_prediction(scheme, fraction, "pos", case, nnunet, reference)
                neg = branch_prediction(scheme, fraction, "neg", case, nnunet, reference)
                # Preserve POS additions; NEG may only remove original nnU-Net voxels.
                fused = pos & ~(nnunet & ~neg)
                patient_dir = folder / case.name
                write_mask(patient_dir / "POS_pred.nii.gz", pos, reference)
                write_mask(patient_dir / "NEG_pred.nii.gz", neg, reference)
                write_mask(patient_dir / "CTV_pred.nii.gz", fused, reference)
                rows.append({
                    "scheme": scheme, "prompt_fraction": fraction, "patient": case.name,
                    "pos_dice": dice(pos, gt), "pos_hd95_mm": hd95(pos, gt, reference.GetSpacing()),
                    "neg_dice": dice(neg, gt), "neg_hd95_mm": hd95(neg, gt, reference.GetSpacing()),
                    "fusion_dice": dice(fused, gt), "fusion_hd95_mm": hd95(fused, gt, reference.GetSpacing()),
                })
            with (folder / "fusion_per_case.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summaries.append({
                "scheme": scheme, "prompt_fraction": fraction, "cases": len(rows),
                "mean_fusion_dice": float(np.mean([float(row["fusion_dice"]) for row in rows])),
                "mean_fusion_hd95_mm": float(np.mean([float(row["fusion_hd95_mm"]) for row in rows])),
            })
    with (OUTPUT_ROOT / "fusion_prompt_fraction_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT_ROOT / "EVALUATION_COMPLETE").touch()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps({
        "selection_source": "validation only", "fractions": FRACTIONS,
        "schemes": SCHEMES, "metrics": ["3D Dice", "3D HD95"],
        "prompt_selection": "deterministic uniform slice selection",
    }, indent=2, default=str) + "\n")
    run_inference()
    evaluate()


if __name__ == "__main__":
    main()
