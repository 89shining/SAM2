#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

# ================= Centralized Paths =================
DEFAULT_DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii")
DEFAULT_EXP_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260423/Train/rule_mask/mask_prompt_3/two_epoch/v1")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_INIT_TRAIN_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/oracle_mask/mask_prompt_2/TrainResult")


def run_cmd(cmd, env=None, cwd=None):
    print("[RUN]", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def main():
    parser = argparse.ArgumentParser(
        description="Run rule-mask two-epoch iterative training then testing (auto-resume enabled)."
    )
    parser.add_argument("--gpu", type=str, default="4", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun nproc per node")
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="torchrun master port. 0 means auto-pick a free port.",
    )
    parser.add_argument("--torchrun", type=str, default="torchrun", help="torchrun executable")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable for test")
    parser.add_argument("--train-script", type=str, default="train.py", help="training script filename")
    parser.add_argument("--test-script", type=str, default="test.py", help="test script filename")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root containing train_nii/test_nii")
    parser.add_argument("--train-subdir", type=str, default="train_nii", help="Train subdir under data-root")
    parser.add_argument("--test-subdir", type=str, default="test_nii", help="Test subdir under data-root")
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT, help="Experiment root containing TrainResult/TestResult")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG, help="SAM2 model config")
    parser.add_argument("--stage1-loss-weight", type=float, default=0.0, help="Stage-1 loss weight passed to train.py")
    parser.add_argument("--stage2-loss-weight", type=float, default=1.0, help="Stage-2 loss weight passed to train.py")
    parser.add_argument(
        "--init-train-output-root",
        type=Path,
        default=DEFAULT_INIT_TRAIN_OUTPUT_ROOT,
        help="External TrainResults root used to auto-resolve best fold checkpoint for initialization.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / args.train_script
    test_script = script_dir / args.test_script

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    if not args.init_train_output_root.exists():
        raise FileNotFoundError(f"init TrainResults root not found: {args.init_train_output_root}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_root = args.data_root / args.train_subdir
    test_root = args.data_root / args.test_subdir
    train_output_root = args.exp_root / "TrainResult"
    test_output_root = args.exp_root / "TestResult"
    master_port = int(args.master_port) if int(args.master_port) > 0 else pick_free_port()

    train_cmd = [
        args.torchrun,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(master_port),
        str(train_script),
        "--train-root",
        str(train_root),
        "--output-root",
        str(train_output_root),
        "--model-cfg",
        str(args.model_cfg),
        "--init-train-output-root",
        str(args.init_train_output_root),
        "--stage1-loss-weight",
        str(args.stage1_loss_weight),
        "--stage2-loss-weight",
        str(args.stage2_loss_weight),
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
        "--model-cfg",
        str(args.model_cfg),
    ]

    print(f"[INFO] Working dir: {script_dir}", flush=True)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(f"[INFO] train_root={train_root}", flush=True)
    print(f"[INFO] test_root={test_root}", flush=True)
    print(f"[INFO] train_output_root={train_output_root}", flush=True)
    print(f"[INFO] test_output_root={test_output_root}", flush=True)
    print(f"[INFO] init_train_output_root={args.init_train_output_root}", flush=True)
    print(f"[INFO] stage1_loss_weight={args.stage1_loss_weight}", flush=True)
    print(f"[INFO] stage2_loss_weight={args.stage2_loss_weight}", flush=True)
    print(f"[INFO] master_port={master_port}", flush=True)
    print("[INFO] Resume policy: ON (always pass --resume)", flush=True)

    print("[1/2] Training (iterative + auto-resume)", flush=True)
    run_cmd(train_cmd, env=env, cwd=str(script_dir))

    print("[2/2] Testing", flush=True)
    run_cmd(test_cmd, env=env, cwd=str(script_dir))

    print("[DONE] Train + Test finished.", flush=True)


if __name__ == "__main__":
    main()
