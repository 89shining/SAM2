#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, env=None, cwd=None):
    print("[RUN]", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run training then testing in iterative mode (SAM2 two-pass)."
    )
    parser.add_argument("--gpu", type=str, default="5", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun nproc per node")
    parser.add_argument("--torchrun", type=str, default="torchrun", help="torchrun executable")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable for test")
    parser.add_argument(
        "--use-torchrun",
        action="store_true",
        help="Use torchrun for training (default: off, so no rendezvous port is needed).",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="train_upper_lower_online_hd95_middle_cv.py",
        help="training script filename",
    )
    parser.add_argument(
        "--test-script",
        type=str,
        default="test_upper_lower_online_hd95_middle.py",
        help="test script filename",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / args.train_script
    test_script = script_dir / args.test_script

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.use_torchrun:
        train_cmd = [
            args.torchrun,
            "--nproc_per_node",
            str(args.nproc_per_node),
            str(train_script),
            "--two-pass-mode",
            "iterative",
        ]
    else:
        train_cmd = [
            args.python,
            str(train_script),
            "--two-pass-mode",
            "iterative",
        ]

    test_cmd = [
        args.python,
        str(test_script),
        "--two-pass-mode",
        "iterative",
    ]

    print(f"[INFO] Working dir: {script_dir}", flush=True)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)

    print("[1/2] Training (iterative)", flush=True)
    run_cmd(train_cmd, env=env, cwd=str(script_dir))

    print("[2/2] Testing (iterative)", flush=True)
    run_cmd(test_cmd, env=env, cwd=str(script_dir))

    print("[DONE] Train + Test finished.", flush=True)


if __name__ == "__main__":
    main()
