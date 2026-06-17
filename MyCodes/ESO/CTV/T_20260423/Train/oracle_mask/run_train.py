#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii")
DEFAULT_EXP_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260423/Train/oracle_mask/k2_to_k10_one_shot")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_PRETRAINED_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
DEFAULT_PROMPT_XLSX = Path("/home/wusi/SAM2/SAM2data/Eso/20260108/Statistics/AAPM/Oracle_Summary.xlsx")


def run_cmd(cmd, env=None, cwd=None):
    print("[RUN]", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, env=env, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed (exit={ret.returncode}): {' '.join(cmd)}")


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def main():
    parser = argparse.ArgumentParser(description="Run oracle one-shot K2..K10 train then test")
    parser.add_argument("--gpu", type=str, default="4")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--torchrun", type=str, default="torchrun")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--test-script", type=str, default="test.py")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-subdir", type=str, default="train_nii")
    parser.add_argument("--test-subdir", type=str, default="test_nii")
    parser.add_argument("--exp-root", type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--pretrained-ckpt", type=Path, default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--prompt-xlsx", type=Path, default=DEFAULT_PROMPT_XLSX)
    parser.add_argument("--ks", type=str, default="2-10")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / args.train_script
    test_script = script_dir / args.test_script

    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    if not args.pretrained_ckpt.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrained_ckpt}")
    if not args.prompt_xlsx.exists():
        raise FileNotFoundError(f"prompt xlsx not found: {args.prompt_xlsx}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_root = args.data_root / args.train_subdir
    test_root = args.data_root / args.test_subdir
    train_output_root = args.exp_root / "TrainResult"
    test_output_root = args.exp_root / "TestResult"
    master_port = int(args.master_port) if int(args.master_port) > 0 else pick_free_port()

    train_cmd = [
        args.torchrun,
        "--nproc_per_node", str(args.nproc_per_node),
        "--master_port", str(master_port),
        str(train_script),
        "--train-root", str(train_root),
        "--output-root", str(train_output_root),
        "--prompt-xlsx", str(args.prompt_xlsx),
        "--model-cfg", str(args.model_cfg),
        "--pretrained-ckpt", str(args.pretrained_ckpt),
        "--ks", str(args.ks),
        "--resume",
    ]

    test_cmd = [
        args.python,
        str(test_script),
        "--test-root", str(test_root),
        "--output-root", str(test_output_root),
        "--train-output-root", str(train_output_root),
        "--prompt-xlsx", str(args.prompt_xlsx),
        "--model-cfg", str(args.model_cfg),
        "--ks", str(args.ks),
    ]

    print(f"[INFO] Working dir: {script_dir}", flush=True)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(f"[INFO] train_root={train_root}", flush=True)
    print(f"[INFO] test_root={test_root}", flush=True)
    print(f"[INFO] train_output_root={train_output_root}", flush=True)
    print(f"[INFO] test_output_root={test_output_root}", flush=True)
    print(f"[INFO] pretrained_ckpt={args.pretrained_ckpt}", flush=True)
    print(f"[INFO] prompt_xlsx={args.prompt_xlsx}", flush=True)
    print(f"[INFO] ks={args.ks}", flush=True)
    print("[INFO] Resume policy: ON", flush=True)

    print("[1/2] Training", flush=True)
    run_cmd(train_cmd, env=env, cwd=str(script_dir))

    print("[2/2] Testing", flush=True)
    run_cmd(test_cmd, env=env, cwd=str(script_dir))

    print("[DONE] Train + Test finished.", flush=True)


if __name__ == "__main__":
    main()
