#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
RECTAL_DATA_ROOT = Path("/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Rectal/20260616_CTV/datanii")
ESO_DATA_ROOT = Path("/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260616_CTV/datanii")
PROJECT_ROOT = Path(os.environ.get("SAM2_PROJECT_ROOT", "/home/intern/ftp/wusi/SAM2")).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from io_utils import DEFAULT_INIT_CKPT, DEFAULT_MODEL_CFG


def run_cmd(cmd: list[str], gpu_id: str | None = None):
    print("[RUN]", " ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    extra_paths = [str(PROJECT_ROOT), str(CURRENT_DIR)]
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([old_pythonpath] if old_pythonpath else []))
    if gpu_id is not None and str(gpu_id).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id).strip()
        print(f"[GPU] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def run_one_dataset(args, data_root: Path, dataset_name: str):
    print("=" * 100, flush=True)
    print(f"[DATASET] {dataset_name}: {data_root}", flush=True)
    print("=" * 100, flush=True)
    train_cmd = [
        sys.executable,
        CURRENT_DIR / "train.py",
        "--data-root",
        data_root,
        "--init-ckpt",
        args.init_ckpt,
        "--model-cfg",
        args.model_cfg,
        "--models",
        args.models,
        "--seed",
        args.seed,
        "--max-epochs",
        args.max_epochs,
        "--patience",
        args.patience,
        "--lr",
        args.lr,
        "--min-lr",
        args.min_lr,
        "--warmup-epochs",
        args.warmup_epochs,
        "--weight-decay",
        args.weight_decay,
        "--grad-clip-norm",
        args.grad_clip_norm,
        "--input-size",
        args.input_size,
        "--window-center",
        args.window_center,
        "--window-width",
        args.window_width,
        "--num-workers",
        args.num_workers,
        "--device",
        args.device,
        "--amp-dtype",
        args.amp_dtype,
        "--lora-r",
        args.lora_r,
        "--lora-alpha",
        args.lora_alpha,
        "--lora-dropout",
        args.lora_dropout,
    ]
    if args.bidirectional_train:
        train_cmd.append("--bidirectional-train")
    if args.forward_backbone_per_frame:
        train_cmd.append("--forward-backbone-per-frame")
    if args.no_resume:
        train_cmd.append("--no-resume")

    run_cmd(train_cmd, gpu_id=args.gpu_id)

    if args.no_test:
        return

    test_cmd = [
        sys.executable,
        CURRENT_DIR / "test.py",
        "--data-root",
        data_root,
        "--init-ckpt",
        args.init_ckpt,
        "--model-cfg",
        args.model_cfg,
        "--models",
        args.models,
        "--test-ks",
        args.test_ks,
        "--folds",
        args.folds,
        "--seed",
        args.seed,
        "--input-size",
        args.input_size,
        "--window-center",
        args.window_center,
        "--window-width",
        args.window_width,
        "--num-workers",
        args.num_workers,
        "--device",
        args.device,
        "--amp-dtype",
        args.amp_dtype,
        "--lora-r",
        args.lora_r,
        "--lora-alpha",
        args.lora_alpha,
        "--lora-dropout",
        args.lora_dropout,
    ]
    if args.forward_backbone_per_frame:
        test_cmd.append("--forward-backbone-per-frame")
    if args.no_save_predictions:
        test_cmd.append("--no-save-predictions")

    run_cmd(test_cmd, gpu_id=args.gpu_id)


def main():
    parser = argparse.ArgumentParser("Run Rectal then Eso prompt-number training/testing")
    parser.add_argument("--single-data-root", type=Path, default=None, help="Run one custom dataset instead of Rectal then Eso.")
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--models", type=str, default="2-6")
    parser.add_argument("--test-ks", type=str, default="2-6")
    parser.add_argument("--folds", type=str, default="0-4")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=None,
        help="GPU id(s) visible to train/test subprocesses, e.g. 0 or 1 or 0,1. Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional-train", action="store_true")
    parser.add_argument("--forward-backbone-per-frame", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="Only run training.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force training from scratch. By default run.py resumes from latest.pth automatically.",
    )
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    if args.single_data_root is not None:
        run_one_dataset(args, args.single_data_root, "Single")
        return

    run_one_dataset(args, RECTAL_DATA_ROOT, "Rectal")
    run_one_dataset(args, ESO_DATA_ROOT, "Eso")


if __name__ == "__main__":
    main()
