#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path


def run_step(step_name: str, script_path: Path, repo_root: Path, env: dict) -> None:
    cmd = [sys.executable, str(script_path)]
    print(f"\n[START] {step_name}")
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    print(f"[DONE] {step_name}")


def main() -> None:
    train_root = Path(__file__).resolve().parent
    repo_root = train_root.parents[4]  # .../SAM2

    steps = [
        (
            "1) oracle_mask/mask_prompt_3/two_epoch TRAIN",
            train_root
            / "oracle_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_middle_cv_two_epoch.py",
        ),
        (
            "1) oracle_mask/mask_prompt_3/two_epoch TEST",
            train_root
            / "oracle_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "test_upper_lower_middle.py",
        ),
        (
            "2) rule_mask/mask_prompt_3/one_epoch TRAIN",
            train_root
            / "rule_mask"
            / "mask_prompt_3"
            / "one_epoch"
            / "train_upper_lower_middle_rule_cv.py",
        ),
        (
            "2) rule_mask/mask_prompt_3/one_epoch TEST",
            train_root
            / "rule_mask"
            / "mask_prompt_3"
            / "one_epoch"
            / "test_upper_lower_middle_rule.py",
        ),
        (
            "3) rule_mask/mask_prompt_3/two_epoch TRAIN",
            train_root
            / "rule_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_middle_rule_cv.py",
        ),
        (
            "3) rule_mask/mask_prompt_3/two_epoch TEST",
            train_root
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
    env["CUDA_VISIBLE_DEVICES"] = "7"
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[INFO] Repo root: {repo_root}")
    print("[INFO] CUDA_VISIBLE_DEVICES=7")
    print("[INFO] Running in strict order: train -> test -> train -> test -> train -> test")

    for step_name, script_path in steps:
        run_step(step_name, script_path, repo_root, env)

    print("\n[ALL DONE] All train/test steps completed successfully.")


if __name__ == "__main__":
    main()
