#!/usr/bin/env python3
"""Training-only physical correction-gate geometry audit.

No model is trained and validation/test cases are never read.  For every branch and
physical disk radius, this script builds deterministic 50% sparse prompts, then
measures the true physical distance from every error voxel to its nearest prompt
voxel.  The generated disk is clipped only by the image FOV, never by the source
error component.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree


CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
PERCENTILES = (25, 50, 75, 90, 95, 99)
DEFAULT_RADII = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)


def pixel_disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def resize_xy(mask: np.ndarray, target: int) -> np.ndarray:
    if mask.shape[1:] == (target, target):
        return mask.astype(bool, copy=False)
    zoom = (1.0, target / mask.shape[1], target / mask.shape[2])
    out = ndimage.zoom(mask.astype(np.uint8), zoom=zoom, order=0, prefilter=False)
    if out.shape[1:] != (target, target):
        fixed = np.zeros((out.shape[0], target, target), dtype=bool)
        h, w = min(target, out.shape[1]), min(target, out.shape[2])
        fixed[:, :h, :w] = out[:, :h, :w] > 0
        return fixed
    return out > 0


def preprocess(raw: np.ndarray, pixel_area_mm2: float) -> np.ndarray:
    """erode2 -> 8-connected top3 -> area>=50 mm2 -> dilate2."""
    eroder = pixel_disk(2)
    dilater = pixel_disk(2)
    result = np.zeros_like(raw, dtype=bool)
    for z in range(raw.shape[0]):
        eroded = ndimage.binary_erosion(raw[z], structure=eroder)
        labels, count = ndimage.label(eroded, structure=CONNECTIVITY_8)
        components: list[tuple[int, int, np.ndarray]] = []
        for label_id in range(1, count + 1):
            component = labels == label_id
            components.append((int(component.sum()), label_id, component))
        components.sort(key=lambda item: (-item[0], item[1]))
        kept = [c for pixels, _, c in components[:3] if pixels * pixel_area_mm2 >= 50.0]
        selected = np.zeros_like(raw[z], dtype=bool)
        for component in kept:
            selected |= component
        if selected.any():
            result[z] = ndimage.binary_dilation(selected, structure=dilater)
    return result


def component_centers(mask: np.ndarray, sy: float, sx: float) -> dict[int, list[tuple[int, int]]]:
    centers: dict[int, list[tuple[int, int]]] = {}
    for z in range(mask.shape[0]):
        labels, count = ndimage.label(mask[z], structure=CONNECTIVITY_8)
        current: list[tuple[int, int]] = []
        for label_id in range(1, count + 1):
            component = labels == label_id
            distance = ndimage.distance_transform_edt(component, sampling=(sy, sx))
            cy, cx = np.unravel_index(int(np.argmax(distance)), distance.shape)
            current.append((int(cy), int(cx)))
        if current:
            centers[z] = current
    return centers


def uniform_half_layers(layers: list[int]) -> list[int]:
    if not layers:
        return []
    # Explicit round-half-up; avoids Python's banker rounding.
    keep = max(1, int(math.floor(0.5 * len(layers) + 0.5)))
    positions = np.floor(np.linspace(0, len(layers) - 1, keep) + 0.5).astype(int)
    selected = [layers[i] for i in np.unique(positions)]
    if len(selected) != keep:
        raise RuntimeError(f"Uniform selection produced {len(selected)} rather than {keep} layers")
    return selected


def make_prompt(
    shape: tuple[int, int, int],
    centers: dict[int, list[tuple[int, int]]],
    selected_layers: list[int],
    sy: float,
    sx: float,
    radius_mm: float,
) -> np.ndarray:
    prompt = np.zeros(shape, dtype=bool)
    ry, rx = int(math.ceil(radius_mm / sy)), int(math.ceil(radius_mm / sx))
    dy, dx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    footprint = (dy * sy) ** 2 + (dx * sx) ** 2 <= radius_mm**2
    fy, fx = np.nonzero(footprint)
    fy, fx = fy - ry, fx - rx
    for z in selected_layers:
        for cy, cx in centers[z]:
            yy, xx = cy + fy, cx + fx
            valid = (yy >= 0) & (yy < shape[1]) & (xx >= 0) & (xx < shape[2])
            prompt[z, yy[valid], xx[valid]] = True
    return prompt


def physical_coordinates(indices_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    return indices_zyx.astype(np.float64) * np.asarray(spacing_zyx, dtype=np.float64)


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, PERCENTILES)
    return {f"p{p}_mm": float(v) for p, v in zip(PERCENTILES, q)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--radii-mm", type=float, nargs="+", default=DEFAULT_RADII)
    ap.add_argument("--coverage-max-mm", type=int, default=50)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_bytes = args.split_json.read_bytes()
    split = json.loads(split_bytes)
    patients = list(split["train"])
    if len(patients) != 99:
        raise RuntimeError(f"Expected 99 training cases, got {len(patients)}")

    case_rows: list[dict[str, object]] = []
    patient_quantile_rows: list[dict[str, object]] = []
    spacing_rows: list[dict[str, object]] = []
    pooled: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    per_case_cache: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    layer_hashes: dict[tuple[str, str], str] = {}

    for patient_index, patient in enumerate(patients, 1):
        case_dir = args.source_root / "Prompt_mask/train" / patient
        ref = sitk.ReadImage(str(case_dir / "CTV.nii.gz"))
        original_x, original_y, original_z = ref.GetSize()
        sx0, sy0, sz = ref.GetSpacing()
        sx = sx0 * original_x / args.input_size
        sy = sy0 * original_y / args.input_size
        spacing_rows.append({
            "patient": patient, "original_x": original_x, "original_y": original_y,
            "slices_z": original_z, "original_sx_mm": sx0, "original_sy_mm": sy0,
            "sz_mm": sz, "resized_sx_mm": sx, "resized_sy_mm": sy,
        })
        gt = resize_xy(sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "CTV.nii.gz"))) > 0, args.input_size)
        nn = resize_xy(sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "nnunet.nii.gz"))) > 0, args.input_size)
        errors = {"pos": gt & ~nn, "neg": nn & ~gt}

        for branch in ("pos", "neg"):
            raw_path = case_dir / f"{branch}_raw.nii.gz"
            raw = resize_xy(sitk.GetArrayFromImage(sitk.ReadImage(str(raw_path))) > 0, args.input_size)
            if not np.array_equal(raw, errors[branch]):
                mismatch = int(np.logical_xor(raw, errors[branch]).sum())
                raise RuntimeError(f"{patient} {branch}: raw error mismatch by {mismatch} voxels")
            base = preprocess(raw, sx * sy)
            centers = component_centers(base, sy, sx)
            valid_layers = sorted(centers)
            selected_layers = uniform_half_layers(valid_layers)
            layer_hash = hashlib.sha256(",".join(map(str, selected_layers)).encode()).hexdigest()
            layer_hashes[(patient, branch)] = layer_hash

            for radius in map(float, args.radii_mm):
                prompt = make_prompt(raw.shape, centers, selected_layers, sy, sx, radius)
                error = errors[branch]
                remain = error & ~prompt  # exact voxel-set E_remain = E & not M_r
                e_count, prompt_count, remain_count = int(error.sum()), int(prompt.sum()), int(remain.sum())
                prompt_empty = not prompt.any()
                unreachable = bool(e_count > 0 and prompt_empty)
                direct_covered = int((error & prompt).sum())
                record: dict[str, object] = {
                    "patient": patient, "branch": branch, "radius_mm": radius,
                    "error_voxels": e_count, "prompt_voxels": prompt_count,
                    "direct_error_voxels": direct_covered, "remain_error_voxels": remain_count,
                    "valid_prompt_layers": len(valid_layers), "selected_prompt_layers": len(selected_layers),
                    "selected_layers_sha256": layer_hash, "prompt_empty": int(prompt_empty),
                    "unreachable_case": int(unreachable), "unreachable_error_voxels": e_count if unreachable else 0,
                }
                distances = np.empty(0, dtype=np.float32)
                all_error_distances = np.empty(0, dtype=np.float32)
                if not prompt_empty and e_count > 0:
                    pcoords = physical_coordinates(np.argwhere(prompt), (sz, sy, sx))
                    tree = cKDTree(pcoords)
                    ecoords = physical_coordinates(np.argwhere(error), (sz, sy, sx))
                    all_error_distances = tree.query(ecoords, k=1, workers=-1)[0].astype(np.float32)
                    if remain_count > 0:
                        rcoords = physical_coordinates(np.argwhere(remain), (sz, sy, sx))
                        distances = tree.query(rcoords, k=1, workers=-1)[0].astype(np.float32)
                        pooled[(branch, radius)].append(distances)
                        pq = percentile_dict(distances)
                        record.update(pq)
                        patient_quantile_rows.append({"patient": patient, "branch": branch, "radius_mm": radius, **pq})
                case_rows.append(record)
                per_case_cache[(branch, radius)].append({
                    "error_voxels": e_count, "direct_error_voxels": direct_covered,
                    "remain_error_voxels": remain_count, "prompt_empty": prompt_empty,
                    "unreachable": unreachable, "remain_distances": distances,
                    "all_error_distances": all_error_distances,
                })
        print(f"[{patient_index:02d}/99] {patient}", flush=True)

    pooled_rows: list[dict[str, object]] = []
    balanced_rows: list[dict[str, object]] = []
    unreachable_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    coverage_points: list[float] = [float(x) for x in range(args.coverage_max_mm + 1)] + [math.inf]

    for branch in ("pos", "neg"):
        for radius in map(float, args.radii_mm):
            key = (branch, radius)
            arrays = pooled[key]
            joined = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
            pooled_rows.append({
                "branch": branch, "radius_mm": radius, "finite_remain_voxels": int(joined.size),
                **(percentile_dict(joined) if joined.size else {f"p{p}_mm": "" for p in PERCENTILES}),
            })
            qrows = [r for r in patient_quantile_rows if r["branch"] == branch and r["radius_mm"] == radius]
            for p in PERCENTILES:
                vals = np.asarray([r[f"p{p}_mm"] for r in qrows], dtype=float)
                balanced_rows.append({
                    "branch": branch, "radius_mm": radius, "case_quantile": f"p{p}",
                    "eligible_cases": len(vals), "mean_mm": float(vals.mean()) if len(vals) else "",
                    "median_mm": float(np.median(vals)) if len(vals) else "",
                    "iqr25_mm": float(np.percentile(vals, 25)) if len(vals) else "",
                    "iqr75_mm": float(np.percentile(vals, 75)) if len(vals) else "",
                })

            cases = per_case_cache[key]
            total_error = sum(int(c["error_voxels"]) for c in cases)
            total_unreachable = sum(int(c["error_voxels"]) for c in cases if c["unreachable"])
            unreachable_rows.append({
                "branch": branch, "radius_mm": radius, "cases": len(cases),
                "unreachable_cases": sum(int(c["unreachable"]) for c in cases),
                "unreachable_error_voxels": total_unreachable, "total_error_voxels": total_error,
                "unreachable_error_fraction": total_unreachable / total_error if total_error else 0.0,
                "zero_error_cases": sum(int(c["error_voxels"] == 0) for c in cases),
                "fully_direct_covered_cases": sum(int(c["error_voxels"] > 0 and c["remain_error_voxels"] == 0) for c in cases),
            })

            for m in coverage_points:
                reachable_num = reachable_den = overall_num = overall_den = 0
                reachable_case_values: list[float] = []
                overall_case_values: list[float] = []
                for c in cases:
                    ecount, rcount = int(c["error_voxels"]), int(c["remain_error_voxels"])
                    if ecount:
                        overall_den += ecount
                        if c["prompt_empty"]:
                            covered_all = 0
                        else:
                            covered_all = int(np.count_nonzero(c["all_error_distances"] <= m))
                        overall_num += covered_all
                        overall_case_values.append(covered_all / ecount)
                    if rcount > 0 and not c["prompt_empty"]:
                        covered_remain = int(np.count_nonzero(c["remain_distances"] <= m))
                        reachable_num += covered_remain
                        reachable_den += rcount
                        reachable_case_values.append(covered_remain / rcount)
                def stats(values: list[float]) -> tuple[object, object, object, object]:
                    if not values:
                        return "", "", "", ""
                    a = np.asarray(values)
                    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))
                rmean, rmed, rq25, rq75 = stats(reachable_case_values)
                omean, omed, oq25, oq75 = stats(overall_case_values)
                coverage_rows.append({
                    "branch": branch, "radius_mm": radius, "gate_mm": "inf" if math.isinf(m) else m,
                    "reachable_voxel_coverage": reachable_num / reachable_den if reachable_den else "",
                    "reachable_patient_mean": rmean, "reachable_patient_median": rmed,
                    "reachable_patient_iqr25": rq25, "reachable_patient_iqr75": rq75,
                    "reachable_eligible_cases": len(reachable_case_values),
                    "overall_voxel_coverage": overall_num / overall_den if overall_den else "",
                    "overall_patient_mean": omean, "overall_patient_median": omed,
                    "overall_patient_iqr25": oq25, "overall_patient_iqr75": oq75,
                    "overall_error_cases": len(overall_case_values),
                })

    write_csv(args.output_dir / "spacing_audit.csv", spacing_rows)
    write_csv(args.output_dir / "per_case_geometry.csv", case_rows)
    write_csv(args.output_dir / "per_case_distance_quantiles.csv", patient_quantile_rows)
    write_csv(args.output_dir / "voxel_pooled_distance_quantiles.csv", pooled_rows)
    write_csv(args.output_dir / "patient_balanced_distance_quantiles.csv", balanced_rows)
    write_csv(args.output_dir / "unreachable_error_summary.csv", unreachable_rows)
    write_csv(args.output_dir / "coverage_curve_1mm.csv", coverage_rows)

    spacing_z = np.asarray([float(r["sz_mm"]) for r in spacing_rows])
    summary = {
        "status": "complete",
        "scope": "training_geometry_only",
        "training_cases": len(patients),
        "validation_cases_read": 0,
        "test_cases_read": 0,
        "split_json": str(args.split_json),
        "split_md5": hashlib.md5(split_bytes).hexdigest(),
        "input_size": args.input_size,
        "radii_mm": list(map(float, args.radii_mm)),
        "prompt_fraction": 0.5,
        "selection": "SI-uniform, round-half-up count",
        "preprocessing": "erode2px -> 8-connected -> top3 -> area>=50mm2 -> dilate2px",
        "disk": "physical XY disk, not clipped to error component, clipped only by image FOV",
        "distance": "nearest prompt voxel Euclidean distance using resized (sz,sy,sx) physical coordinates",
        "z_spacing_mm": {
            "min": float(spacing_z.min()), "median": float(np.median(spacing_z)),
            "max": float(spacing_z.max()), "unique_counts": dict(Counter(map(str, spacing_z))),
        },
        "candidate_grid_locked": False,
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
