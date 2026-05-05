#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ====== Put all paths here ======
TRAIN_SCRIPT = Path("train.py")
TEST_SCRIPT = Path("test.py")

TRAIN_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii/train_nii")
TEST_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii/test_nii")
PRETRAINED_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

TRAIN_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260423/nnUNet/mask/TrainResults")
TEST_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260423/nnUNet/mask/TestResults")

# ====== Train hyperparameters ======
NUM_FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 1
NUM_WORKERS = 4
INPUT_SIZE = 1024
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 400.0
LR = 1e-3
WEIGHT_DECAY = 0.05
ETA_MIN_FACTOR = 0.1
SEED = 42
DEVICE = "cuda"
AMP_DTYPE = "bfloat16"
IMAGE_NAME = "image.nii.gz"
MASK_NAME = "CTV.nii.gz"
PROMPT_NAME = "prompt.nii.gz"

# ====== Test options ======
OBJ_ID = 1


def _run(cmd, cwd: Path):
    print("[CMD]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _train_done(output_root: Path) -> bool:
    return (output_root / "cv_summary.csv").exists() and (output_root / "best_fold.txt").exists()


def main():
    parser = argparse.ArgumentParser("Run train.py then test.py with central config")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run testing")
    parser.add_argument("--force-train", action="store_true", help="Force rerun training even if completed files exist")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing train script: {TRAIN_SCRIPT}")
    if not TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Missing test script: {TEST_SCRIPT}")

    if not args.skip_train:
        need_train = args.force_train or (not _train_done(TRAIN_OUTPUT_ROOT))
        if need_train:
            TRAIN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            train_cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--train-root", str(TRAIN_ROOT),
                "--output-root", str(TRAIN_OUTPUT_ROOT),
                "--pretrained-ckpt", str(PRETRAINED_CKPT),
                "--model-cfg", MODEL_CFG,
                "--image-name", IMAGE_NAME,
                "--mask-name", MASK_NAME,
                "--prompt-name", PROMPT_NAME,
                "--num-folds", str(NUM_FOLDS),
                "--epochs", str(EPOCHS),
                "--batch-size", str(BATCH_SIZE),
                "--num-workers", str(NUM_WORKERS),
                "--input-size", str(INPUT_SIZE),
                "--window-center", str(WINDOW_CENTER),
                "--window-width", str(WINDOW_WIDTH),
                "--lr", str(LR),
                "--weight-decay", str(WEIGHT_DECAY),
                "--eta-min-factor", str(ETA_MIN_FACTOR),
                "--seed", str(SEED),
                "--device", DEVICE,
                "--amp-dtype", AMP_DTYPE,
            ]
            _run(train_cmd, workdir)
        else:
            print("[INFO] Training already completed, auto-resume by skipping train and moving to test.")

    TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    test_cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--test-root", str(TEST_ROOT),
        "--train-output-root", str(TRAIN_OUTPUT_ROOT),
        "--output-root", str(TEST_OUTPUT_ROOT),
        "--model-cfg", MODEL_CFG,
        "--img-name", IMAGE_NAME,
        "--gt-name", MASK_NAME,
        "--prompt-name", PROMPT_NAME,
        "--obj-id", str(OBJ_ID),
        "--window-center", str(WINDOW_CENTER),
        "--window-width", str(WINDOW_WIDTH),
        "--device", DEVICE,
    ]
    _run(test_cmd, workdir)

    print("[DONE] Train/Test pipeline finished.")


if __name__ == "__main__":
    main()
