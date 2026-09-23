#!/usr/bin/env python3
"""Create validation heatmaps and per-sigma Dice/HD95/best-m profiles."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--metrics-root',type=Path,required=True); a=p.parse_args(); fig=a.metrics_root/'figures'; fig.mkdir(parents=True,exist_ok=True)
for kind in ('pos','neg'):
    with (a.metrics_root/f'validation_{kind}_grid.csv').open() as f: grid=list(csv.DictReader(f))
    with (a.metrics_root/f'validation_{kind}_profile.csv').open() as f: profile=list(csv.DictReader(f))
    ts=sorted({float(r['sigma_mm']) for r in grid}); ms=[0,5,10,15,20,30,40,60,float('inf')]
    for metric,label in [('mean_dice','Dice'),('mean_hd95_mm','HD95 (mm)')]:
        arr=np.array([[float(next(r[metric] for r in grid if float(r['sigma_mm'])==t and r['gate_mm']==('inf' if np.isinf(m) else str(int(m))))) for m in ms] for t in ts])
        im=plt.imshow(arr,aspect='auto',cmap='viridis'); plt.colorbar(im,label=label); plt.xticks(range(len(ms)),['∞' if np.isinf(m) else str(int(m)) for m in ms]); plt.yticks(range(len(ts)),[str(int(t)) for t in ts]); plt.xlabel('gate m (mm)'); plt.ylabel('sigma (mm)'); plt.title(f'{kind.upper()} validation {label}'); plt.tight_layout(); plt.savefig(fig/f'{kind}_validation_{metric}_heatmap.png',dpi=180); plt.close()
    x=np.array([float(r['sigma_mm']) for r in profile]); d=np.array([float(r['mean_dice']) for r in profile]); h=np.array([float(r['mean_hd95_mm']) for r in profile]); m=[r['selected_gate_mm'] for r in profile]
    for y,name,ylab in [(d,'dice','Mean Dice'),(h,'hd95','Mean HD95 (mm)')]:
        plt.plot(x,y,'o-'); [plt.annotate(f'm={q}',(xx,yy),xytext=(0,6),textcoords='offset points',ha='center') for xx,yy,q in zip(x,y,m)]; plt.xlabel('skeleton sigma (mm)'); plt.ylabel(ylab); plt.title(f'{kind.upper()} validation best-gate profile'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(fig/f'{kind}_validation_{name}_profile.png',dpi=180); plt.close()
    plt.plot(x,[np.inf if q=='inf' else float(q) for q in m],'o-'); plt.xlabel('skeleton sigma (mm)'); plt.ylabel('selected gate m (mm)'); plt.title(f'{kind.upper()} validation best m by sigma'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(fig/f'{kind}_validation_best_m.png',dpi=180); plt.close()
