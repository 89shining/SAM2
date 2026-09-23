#!/usr/bin/env python3
"""Fail-closed audit for source skeleton prompts; no source file is modified."""
from __future__ import annotations
import argparse
from pathlib import Path
import SimpleITK as sitk

def same(a, b):
    return a.GetSize()==b.GetSize() and a.GetSpacing()==b.GetSpacing() and a.GetOrigin()==b.GetOrigin() and a.GetDirection()==b.GetDirection()

p=argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, required=True); p.add_argument('--marker', type=Path, required=True); a=p.parse_args()
a.marker.unlink(missing_ok=True)
checked=0
for subset in ('train','test'):
    for case in sorted((a.data_root/subset).glob('p_*')):
        ref=sitk.ReadImage(str(case/'image.nii.gz'))
        for kind in ('pos','neg'):
            sk=case/f'{kind}_skeleton.nii.gz'
            if not sk.is_file(): raise FileNotFoundError(sk)
            if not same(ref, sitk.ReadImage(str(sk))): raise ValueError(f'geometry mismatch: {sk}')
            checked += 1
a.marker.parent.mkdir(parents=True, exist_ok=True); a.marker.touch()
print(f'PASS: {checked} skeleton volumes; marker={a.marker}')
