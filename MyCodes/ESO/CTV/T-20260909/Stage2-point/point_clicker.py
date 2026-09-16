"""Deterministic/stochastic 3-D RITM-style correction-point sampling."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CorrectionPoint:
    """One point on the preprocessed discrete [z, y, x] voxel grid."""

    z: int
    y: int
    x: int
    label: int  # +1 for FN / foreground, -1 for FP / background
    error_type: Literal["FN", "FP"]
    component_voxels: int
    component_mm3: float


_CONN26 = np.ones((3, 3, 3), dtype=np.uint8)


def _largest_labeled_component(
    mask: np.ndarray, error_type: str,
) -> tuple[tuple[int, tuple[int, int, int], str, int], np.ndarray] | None:
    """Find the largest 26-connected component without materializing every mask."""
    labels, count = ndimage.label(mask.astype(bool), structure=_CONN26)
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel(), minlength=int(count) + 1)
    sizes[0] = 0
    largest = int(sizes.max())
    best: tuple[int, tuple[int, int, int], str, int] | None = None
    # Only tied maxima require a full-volume equality mask to preserve the
    # exact former lexicographic tie-break. Typical calls scan one component.
    for index in np.flatnonzero(sizes == largest):
        first = tuple(int(value) for value in np.argwhere(labels == index)[0])
        candidate = (-largest, first, error_type, int(index))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best, labels


def _largest_error_component(
    gt_zyx: np.ndarray,
    pred_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    exclude_slices: Sequence[int] = (),
) -> tuple[np.ndarray, str] | None:
    """Select the largest FN/FP component, deterministic under exact ties."""
    gt = np.asarray(gt_zyx, dtype=bool)
    pred = np.asarray(pred_zyx, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"GT/prediction shape mismatch: {gt.shape} vs {pred.shape}")
    voxel_mm3 = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    winner: tuple[int, tuple[int, int, int], str, int] | None = None
    winning_labels: np.ndarray | None = None
    excluded = np.zeros(gt.shape[0], dtype=bool)
    for z in exclude_slices:
        z = int(z)
        if not 0 <= z < gt.shape[0]:
            raise IndexError(f"Excluded slice {z} is outside Z={gt.shape[0]}")
        excluded[z] = True
    for error_type, error in (("FN", gt & ~pred), ("FP", pred & ~gt)):
        error = error.copy()
        error[excluded] = False
        found = _largest_labeled_component(error, error_type)
        if found is None:
            continue
        candidate, labels = found
        if winner is None or candidate < winner:
            winner, winning_labels = candidate, labels
    if winner is None or winning_labels is None:
        return None
    # Physical volume is count * identical patient voxel volume; count is exact.
    # Prefer FN before FP only when volume and location are exactly tied.
    _ = voxel_mm3  # Explicitly document physical-volume equivalence within patient.
    _, _, error_type, label = winner
    return winning_labels == label, error_type


def sample_correction_point(
    gt_zyx: np.ndarray,
    pred_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    mode: Literal["train", "validation"],
    rng: random.Random | None = None,
    exclude_slices: Sequence[int] = (),
) -> CorrectionPoint | None:
    """Return one correction click, or ``None`` if the prediction is exact.

    Training samples uniformly from the most-internal ceil(25%) voxels. Validation
    picks the EDT maximum, resolving equal distances by smallest (z, y, x).
    """
    selected = _largest_error_component(
        gt_zyx, pred_zyx, spacing_zyx, exclude_slices=exclude_slices
    )
    if selected is None:
        return None
    component, error_type = selected
    count = int(component.sum())
    distances = ndimage.distance_transform_edt(
        component, sampling=tuple(float(value) for value in spacing_zyx)
    )
    coords = np.argwhere(component)
    values = distances[component]
    if mode == "validation":
        maximum = float(values.max())
        tied = coords[np.isclose(values, maximum)]
        z, y, x = (int(value) for value in tied[0])
    elif mode == "train":
        if rng is None:
            raise ValueError("Training correction sampling requires an explicit RNG")
        keep = max(1, int(math.ceil(0.25 * count)))
        # Stable ordering makes the same seeded RNG reproducible across resumes.
        order = np.argsort(-values, kind="stable")
        candidates = coords[order[:keep]]
        z, y, x = (int(value) for value in candidates[rng.randrange(len(candidates))])
    else:
        raise ValueError(f"Unknown correction-point mode: {mode}")
    return CorrectionPoint(
        z=z, y=y, x=x,
        label=1 if error_type == "FN" else -1,
        error_type=error_type,
        component_voxels=count,
        component_mm3=float(count * np.prod(np.asarray(spacing_zyx, dtype=np.float64))),
    )

