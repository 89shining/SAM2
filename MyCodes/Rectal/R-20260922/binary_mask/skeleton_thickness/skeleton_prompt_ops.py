"""2-D skeleton segment selection and physical thickening for R-20260922.

Input arrays use SimpleITK/NumPy order [Z, Y, X]; spacing remains (sx, sy, sz).
No operation in this module uses GT, raw error masks, or 3-D morphology.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import random

import numpy as np
from scipy import ndimage


_EIGHT = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Worker-local cache keyed by the stable in-memory skeleton array.  It caches
# only deterministic graph/diameter work; length fraction and segment start
# remain newly sampled on every training call.
_SEGMENT_CACHE: dict[tuple[int, float, float], tuple[tuple[int, tuple[tuple[int, int], ...], np.ndarray], ...]] = {}


def _adjacency(points: np.ndarray, sx: float, sy: float) -> dict[tuple[int, int], list[tuple[tuple[int, int], float]]]:
    nodes = {tuple(map(int, p)) for p in points}
    graph = {node: [] for node in nodes}
    for y, x in nodes:
        for dy, dx in _EIGHT:
            nxt = (y + dy, x + dx)
            if nxt in nodes:
                graph[(y, x)].append((nxt, sx if dy == 0 else sy if dx == 0 else math.hypot(sx, sy)))
    return graph


def _dijkstra(graph, source):
    dist = {source: 0.0}; prev = {}
    queue = [(0.0, source)]
    while queue:
        current, node = heapq.heappop(queue)
        if current != dist[node]:
            continue
        for nxt, weight in graph[node]:
            candidate = current + weight
            if candidate < dist.get(nxt, math.inf):
                dist[nxt] = candidate; prev[nxt] = node
                heapq.heappush(queue, (candidate, nxt))
    return dist, prev


def _longest_path(graph):
    """Deterministic weighted diameter path; robust to loops and branch points."""
    endpoints = sorted(node for node, edges in graph.items() if len(edges) == 1)
    seed = endpoints[0] if endpoints else min(graph)
    first, _ = _dijkstra(graph, seed)
    a = max(first, key=lambda node: (first[node], node))
    second, prev = _dijkstra(graph, a)
    b = max(second, key=lambda node: (second[node], node))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    coordinates = [0.0]
    for left, right in zip(path[:-1], path[1:]):
        weight = next(w for n, w in graph[left] if n == right)
        coordinates.append(coordinates[-1] + weight)
    return path, np.asarray(coordinates, dtype=np.float64)


def _prepared_paths(skeleton_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]):
    """Build deterministic component diameter paths once per cached case."""
    sx, sy, _ = spacing_xyz
    key = (skeleton_zyx.shape, hashlib.blake2b(
        np.ascontiguousarray(skeleton_zyx).view(np.uint8), digest_size=16
    ).digest(), float(sx), float(sy))
    cached = _SEGMENT_CACHE.get(key)
    if cached is not None:
        return cached
    structure = np.ones((3, 3), dtype=np.uint8)
    prepared = []
    for z, frame in enumerate(skeleton_zyx.astype(bool, copy=False)):
        labels, count = ndimage.label(frame, structure=structure)
        for component_id in range(1, count + 1):
            points = np.argwhere(labels == component_id)
            if len(points) == 1:
                prepared.append((z, (tuple(map(int, points[0])),), np.asarray([0.0], dtype=np.float64)))
                continue
            path, coordinate = _longest_path(_adjacency(points, sx, sy))
            prepared.append((z, tuple(path), coordinate))
    cached = tuple(prepared)
    _SEGMENT_CACHE[key] = cached
    return cached


def select_contiguous_segments(skeleton_zyx: np.ndarray, spacing_xyz: tuple[float, float, float], fraction: float, mode: str) -> np.ndarray:
    """Keep a contiguous geodesic segment per 2-D skeleton component.

    Graph construction is cached, while random length and location selection
    deliberately occur on every call.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    if mode not in {"random", "center"}:
        raise ValueError(f"Unsupported segment mode: {mode}")
    output = np.zeros_like(skeleton_zyx, dtype=bool)
    for z, path, coordinate in _prepared_paths(skeleton_zyx, spacing_xyz):
        total = float(coordinate[-1])
        if total <= 0.0:
            y, x = path[0]
            output[z, y, x] = True
            continue
        retained = total * fraction
        start = (total - retained) * 0.5 if mode == "center" else random.uniform(0.0, total - retained)
        stop = start + retained
        selected = [node for node, value in zip(path, coordinate) if start - 1e-8 <= value <= stop + 1e-8]
        if not selected:
            selected = [path[int(np.argmin(np.abs(coordinate - (start + stop) * 0.5)))]]
        for y, x in selected:
            output[z, y, x] = True
    return output

def thicken_2d(skeleton_zyx: np.ndarray, spacing_xyz: tuple[float, float, float], thickness_mm: float) -> np.ndarray:
    """Physical 2-D dilation around skeleton; t=0 returns the skeleton itself."""
    if thickness_mm < 0:
        raise ValueError("thickness_mm must be non-negative")
    if thickness_mm == 0:
        return skeleton_zyx.astype(bool, copy=True)
    sx, sy, _ = spacing_xyz
    output = np.zeros_like(skeleton_zyx, dtype=bool)
    for z, frame in enumerate(skeleton_zyx.astype(bool, copy=False)):
        if frame.any():
            # frame is foreground; EDT of its complement returns distance to skeleton in [Y, X].
            output[z] = ndimage.distance_transform_edt(~frame, sampling=(sy, sx)) <= thickness_mm
    return output


def build_skeleton_prompt(skeleton_zyx: np.ndarray, spacing_xyz: tuple[float, float, float], thickness_mm: float, length_fraction: float, mode: str) -> np.ndarray:
    return thicken_2d(select_contiguous_segments(skeleton_zyx, spacing_xyz, length_fraction, mode), spacing_xyz, thickness_mm)
