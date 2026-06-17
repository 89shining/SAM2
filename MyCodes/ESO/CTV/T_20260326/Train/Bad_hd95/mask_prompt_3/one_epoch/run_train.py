#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ================= Centralized Paths =================
DEFAULT_DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii")
DEFAULT_EXP_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/BadHD95_slice/mask_prompt_3/one_epoch")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SELECTOR_TRAIN_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/oracle_mask/mask_prompt_2/TrainResult")
DEFAULT_PRETRAINED_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")


def run_cmd(cmd, env=None, cwd=None):
    print("[RUN]", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run bad_hd95 one-epoch training: selector picks worst middle, main model trains from initial ckpt."
    )
    parser.add_argument("--gpu", type=str, default="5", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun nproc per node")
    parser.add_argument("--torchrun", type=str, default="torchrun", help="torchrun executable")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable for test")
    parser.add_argument("--train-script", type=str, default="train.py", help="training script filename")
    parser.add_argument("--test-script", type=str, default="test.py", help="test script filename")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root containing train_nii/test_nii")
    parser.add_argument("--train-subdir", type=str, default="train_nii", help="Train subdir under data-root")
    parser.add_argument("--test-subdir", type=str, default="test_nii", help="Test subdir under data-root")
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT, help="Experiment root containing TrainResult/TestResult")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG, help="SAM2 model config")
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=None,
        help="Optional explicit TrainResult path. If set, overrides exp-root/TrainResult.",
    )
    parser.add_argument(
        "--test-output-root",
        type=Path,
        default=None,
        help="Optional explicit TestResult path. If set, overrides exp-root/TestResult.",
    )
    parser.add_argument(
        "--selector-train-output-root",
        type=Path,
        default=DEFAULT_SELECTOR_TRAIN_OUTPUT_ROOT,
        help="External first-round TrainResults root used by selector model to choose worst middle slice.",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=DEFAULT_PRETRAINED_CKPT,
        help="Initial checkpoint used by the one-epoch main model.",
    )
    parser.add_argument(
        "--finetuned-ckpt",
        type=Path,
        default=None,
        help="Optional explicit checkpoint for testing. If omitted, test.py auto-resolves from train-output-root.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / args.train_script
    test_script = script_dir / args.test_script

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    if not args.selector_train_output_root.exists():
        raise FileNotFoundError(f"selector TrainResults root not found: {args.selector_train_output_root}")
    if not args.pretrained_ckpt.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrained_ckpt}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_root = args.data_root / args.train_subdir
    test_root = args.data_root / args.test_subdir
    train_output_root = args.train_output_root if args.train_output_root is not None else (args.exp_root / "TrainResult")
    test_output_root = args.test_output_root if args.test_output_root is not None else (args.exp_root / "TestResult")

    train_cmd = [
        args.torchrun,
        "--nproc_per_node",
        str(args.nproc_per_node),
        str(train_script),
        "--train-root",
        str(train_root),
        "--output-root",
        str(train_output_root),
        "--model-cfg",
        str(args.model_cfg),
        "--selector-train-output-root",
        str(args.selector_train_output_root),
        "--pretrained-ckpt",
        str(args.pretrained_ckpt),
        "--resume",
    ]

    test_cmd = [
        args.python,
        str(test_script),
        "--test-root",
        str(test_root),
        "--output-root",
        str(test_output_root),
        "--train-output-root",
        str(train_output_root),
        "--selector-train-output-root",
        str(args.selector_train_output_root),
        "--model-cfg",
        str(args.model_cfg),
    ]
    if args.finetuned_ckpt is not None:
        test_cmd.extend(["--finetuned-ckpt", str(args.finetuned_ckpt)])

    print(f"[INFO] Working dir: {script_dir}", flush=True)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(f"[INFO] train_root={train_root}", flush=True)
    print(f"[INFO] test_root={test_root}", flush=True)
    print(f"[INFO] train_output_root={train_output_root}", flush=True)
    print(f"[INFO] test_output_root={test_output_root}", flush=True)
    print(f"[INFO] selector_train_output_root={args.selector_train_output_root}", flush=True)
    print(f"[INFO] pretrained_ckpt={args.pretrained_ckpt}", flush=True)
    if args.finetuned_ckpt is not None:
        print(f"[INFO] finetuned_ckpt={args.finetuned_ckpt}", flush=True)
    print("[INFO] Resume policy: ON (always pass --resume)", flush=True)

    print("[1/2] Training (iterative + auto-resume)", flush=True)
    run_cmd(train_cmd, env=env, cwd=str(script_dir))

    print("[2/2] Testing", flush=True)
    run_cmd(test_cmd, env=env, cwd=str(script_dir))

    print("[DONE] Train + Test finished.", flush=True)


if __name__ == "__main__":
    main()
