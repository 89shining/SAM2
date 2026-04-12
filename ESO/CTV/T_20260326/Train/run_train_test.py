#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    # Support both .../SAM2/ESO/... and possible relocated script paths.
    for p in [start] + list(start.parents):
        if p.name == "SAM2" and (p / "ESO").is_dir():
            return p
    raise RuntimeError(f"Cannot locate SAM2 repo root from: {start}")


def run_step(step_name: str, script_path: Path, repo_root: Path, env: dict, use_torchrun: bool, nproc_per_node: int) -> None:
    if use_torchrun:
        cmd = ["torchrun", f"--nproc_per_node={nproc_per_node}", str(script_path)]
    else:
        cmd = [sys.executable, str(script_path)]
    print(f"\n[START] {step_name}")
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    print(f"[DONE] {step_name}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)

    # Compatible with script placed in:
    # 1) .../T_20260326/Train/run_train_test.py
    # 2) .../T_20260326/run_train_test.py
    if (script_dir / "oracle_mask").is_dir():
        train_root = script_dir
    elif (script_dir / "Train" / "oracle_mask").is_dir():
        train_root = script_dir / "Train"
    else:
        train_root = repo_root / "ESO" / "CTV" / "T_20260326" / "Train"

    steps = [
        (
            "1) T_20260410/Bad_hd95/Train TRAIN",
            repo_root
            / "ESO"
            / "CTV"
            / "T_20260410"
            / "Bad_hd95"
            / "Train"
            / "train_upper_lower_online_hd95_middle_cv.py",
        ),
        (
            "1) T_20260410/Bad_hd95/Train TEST",
            repo_root
            / "ESO"
            / "CTV"
            / "T_20260410"
            / "Bad_hd95"
            / "Train"
            / "test_upper_lower_online_hd95_middle.py",
        ),
        (
            "2) oracle_mask/mask_prompt_3/two_epoch TRAIN",
            repo_root
            / "ESO"
            / "CTV"
            / "T_20260326"
            / "Train"
            / "oracle_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_middle_cv_two_epoch.py",
        ),
        (
            "2) oracle_mask/mask_prompt_3/two_epoch TEST",
            repo_root
            / "ESO"
            / "CTV"
            / "T_20260326"
            / "Train"
            / "oracle_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "test_upper_lower_middle.py",
        ),
        (
            "3) rule_mask/mask_prompt_3/two_epoch TRAIN",
            repo_root
            / "ESO"
            / "CTV"
            / "T_20260326"
            / "Train"
            / "rule_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_middle_rule_cv.py",
        ),
        (
            "3) rule_mask/mask_prompt_3/two_epoch TEST",
            repo_root
            / "ESO"
            / "CTV"
            / "T_20260326"
            / "Train"
            / "rule_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "test_upper_lower_middle_rule.py",
        ),
    ]

    missing = [str(p) for _, p in steps if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing script(s):\n" + "\n".join(missing))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    env["PYTHONUNBUFFERED"] = "1"
    nproc_per_node = 3

    print(f"[INFO] Repo root: {repo_root}")
    print("[INFO] CUDA_VISIBLE_DEVICES=5,6,7")
    print(f"[INFO] torchrun --nproc_per_node={nproc_per_node} for TRAIN steps")
    print("[INFO] Running in strict order:")
    print("[INFO]   1) Bad_hd95 train -> test")
    print("[INFO]   2) oracle_mask/two_epoch train -> test")
    print("[INFO]   3) rule_mask/two_epoch train -> test")

    for step_name, script_path in steps:
        use_torchrun = " TRAIN" in step_name
        run_step(step_name, script_path, repo_root, env, use_torchrun=use_torchrun, nproc_per_node=nproc_per_node)

    print("\n[ALL DONE] All train/test steps completed successfully.")


if __name__ == "__main__":
    main()
