#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd: Path):
    print("[RUN]", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run v4 and v5 pipelines sequentially (both support auto-resume)."
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--gpu", type=str, default="5", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun nproc per node")
    parser.add_argument("--master-port", type=int, default=0, help="torchrun master port (0 means auto)")
    parser.add_argument("--torchrun", type=str, default="torchrun", help="torchrun executable")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii"),
        help="Dataset root containing train_nii/test_nii",
    )
    parser.add_argument("--train-subdir", type=str, default="train_nii", help="Train subdir under data-root")
    parser.add_argument("--test-subdir", type=str, default="test_nii", help="Test subdir under data-root")
    parser.add_argument(
        "--exp-root-base",
        type=Path,
        default=Path("/home/wusi/SAM2/SAM2data/Eso/20260423/Try_rule_mask/mask_prompt_3/two_epoch"),
        help="Base experiment root; script will use <base>/v4 and <base>/v5",
    )
    parser.add_argument("--model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--init-ckpt",
        type=Path,
        default=Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt"),
        help="Initialization checkpoint path (SAM2 original)",
    )
    parser.add_argument(
        "--init-train-output-root",
        type=Path,
        default=None,
        help="Optional fallback for init checkpoint resolution",
    )
    parser.add_argument("--v4-stage1-loss-weight", type=float, default=0.5)
    parser.add_argument("--v4-stage2-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--v5-seed-prompt-mode",
        type=str,
        choices=["single", "bounds"],
        default="single",
        help="Seed prompt mode for v5",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["both", "v4", "v5"],
        default="both",
        help="Run both versions or only one",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    v4_run = script_dir / "v4" / "run_train.py"
    v5_run = script_dir / "v5" / "run_train.py"
    if not v4_run.exists():
        raise FileNotFoundError(f"v4 runner not found: {v4_run}")
    if not v5_run.exists():
        raise FileNotFoundError(f"v5 runner not found: {v5_run}")

    common = [
        "--gpu",
        str(args.gpu),
        "--nproc-per-node",
        str(args.nproc_per_node),
        "--master-port",
        str(args.master_port),
        "--torchrun",
        str(args.torchrun),
        "--python",
        str(args.python),
        "--data-root",
        str(args.data_root),
        "--train-subdir",
        str(args.train_subdir),
        "--test-subdir",
        str(args.test_subdir),
        "--model-cfg",
        str(args.model_cfg),
        "--init-ckpt",
        str(args.init_ckpt),
    ]
    if args.init_train_output_root is not None:
        common.extend(["--init-train-output-root", str(args.init_train_output_root)])

    if args.only in ("both", "v4"):
        cmd_v4 = [
            str(args.python),
            str(v4_run),
            *common,
            "--exp-root",
            str(args.exp_root_base / "v4"),
            "--stage1-loss-weight",
            str(args.v4_stage1_loss_weight),
            "--stage2-loss-weight",
            str(args.v4_stage2_loss_weight),
        ]
        print("\n[PIPELINE] v4 (auto-resume enabled in v4/run_train.py)", flush=True)
        run_cmd(cmd_v4, cwd=script_dir)

    if args.only in ("both", "v5"):
        cmd_v5 = [
            str(args.python),
            str(v5_run),
            *common,
            "--exp-root",
            str(args.exp_root_base / "v5"),
            "--seed-prompt-mode",
            str(args.v5_seed_prompt_mode),
        ]
        print("\n[PIPELINE] v5 (auto-resume enabled in v5/run_train.py)", flush=True)
        run_cmd(cmd_v5, cwd=script_dir)

    print("\n[DONE] Requested pipelines finished.", flush=True)


if __name__ == "__main__":
    main()
