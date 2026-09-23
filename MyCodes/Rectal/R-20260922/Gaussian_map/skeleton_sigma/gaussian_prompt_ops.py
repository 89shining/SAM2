"""Physical 2-D Gaussian prompt construction from existing axial skeletons."""
from __future__ import annotations
import hashlib
import heapq, random
import numpy as np
from scipy import ndimage

# Worker-local deterministic skeleton graph cache. Random segment selection is
# intentionally not cached.
_SEGMENT_CACHE = {}

def _graph(points, sx, sy):
    pts={tuple(map(int,p)) for p in points}; g={p:[] for p in pts}
    for y,x in pts:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0 or (y+dy,x+dx) not in pts: continue
                g[(y,x)].append(((y+dy,x+dx), float(np.hypot(dx*sx,dy*sy))))
    return g
def _dist(g,start):
    d={start:0.}; parent={}; q=[(0.,start)]
    while q:
        v,u=heapq.heappop(q)
        if v!=d[u]: continue
        for w,c in g[u]:
            z=v+c
            if z<d.get(w,float('inf')): d[w]=z; parent[w]=u; heapq.heappush(q,(z,w))
    return d,parent
def _diameter_path(g):
    seed=min(g); d,_=_dist(g,seed); a=max(d,key=d.get); d,parent=_dist(g,a); b=max(d,key=d.get); path=[b]
    while path[-1]!=a: path.append(parent[path[-1]])
    path=path[::-1]; length=[0.]
    for u,v in zip(path,path[1:]): length.append(length[-1]+next(c for w,c in g[u] if w==v))
    return path,np.asarray(length),length[-1]
def _prepared_paths(skeleton_zyx, spacing_xyz):
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    key = (skeleton_zyx.shape, hashlib.blake2b(
        np.ascontiguousarray(skeleton_zyx).view(np.uint8), digest_size=16
    ).digest(), sx, sy)
    cached = _SEGMENT_CACHE.get(key)
    if cached is not None:
        return cached
    prepared = []
    for z, frame in enumerate(skeleton_zyx.astype(bool, copy=False)):
        labels, n = ndimage.label(frame, structure=np.ones((3, 3), bool))
        for lab in range(1, n + 1):
            pts = np.argwhere(labels == lab)
            if len(pts) == 1:
                prepared.append((z, (tuple(map(int, pts[0])),), np.asarray([0.], dtype=np.float64)))
            else:
                path, pos, _ = _diameter_path(_graph(pts, sx, sy))
                prepared.append((z, tuple(path), pos))
    cached = tuple(prepared)
    _SEGMENT_CACHE[key] = cached
    return cached

def contiguous_segment(skeleton_zyx, spacing_xyz, fraction, mode):
    """One weighted-geodesic segment per component; only static paths cache."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'fraction must lie in (0, 1], got {fraction}')
    if mode not in {'random', 'center'}:
        raise ValueError(f'Unsupported segment mode: {mode}')
    out = np.zeros_like(skeleton_zyx, dtype=bool)
    for z, path, pos in _prepared_paths(skeleton_zyx, spacing_xyz):
        total = float(pos[-1])
        if total <= 0.0:
            out[z, path[0][0], path[0][1]] = True
            continue
        keep = total * fraction
        start = (total - keep) / 2 if mode == 'center' else random.uniform(0, total - keep)
        end = start + keep
        selected = [point for point, d in zip(path, pos) if start - 1e-9 <= d <= end + 1e-9]
        if not selected:
            selected = [path[int(np.argmin(np.abs(pos - (start + end) / 2)) )]]
        for y, x in selected:
            out[z, y, x] = True
    return out
def gaussian_map(skeleton_zyx, spacing_xyz, sigma_mm, fraction, mode):
    """Return float32 [Z,Y,X] map, exp(-d²/(2σ²)), hard-truncated at 3σ."""
    if sigma_mm<=0: raise ValueError('sigma_mm must be positive')
    seg=contiguous_segment(skeleton_zyx,spacing_xyz,fraction,mode); sx,sy=float(spacing_xyz[0]),float(spacing_xyz[1]); out=np.zeros(seg.shape,np.float32); cut=3*float(sigma_mm)
    for z,frame in enumerate(seg):
        if not frame.any(): continue
        d=ndimage.distance_transform_edt(~frame,sampling=(sy,sx)); value=np.exp(-(d*d)/(2*sigma_mm*sigma_mm)); value[d>cut]=0.; out[z]=value.astype(np.float32)
    return out
