#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd: Path):
    print("[RUN]", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, cwd=str(cwd))
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed (exit={ret.returncode}): {' '.join(cmd)}")


def build_common_args(args):
    out = []
    if args.gpu is not None:
        out += ["--gpu", str(args.gpu)]
    if args.nproc_per_node is not None:
        out += ["--nproc-per-node", str(args.nproc_per_node)]
    if args.ks is not None:
        out += ["--ks", str(args.ks)]
    if args.model_cfg is not None:
        out += ["--model-cfg", str(args.model_cfg)]
    if args.data_root is not None:
        out += ["--data-root", str(args.data_root)]
    if args.train_subdir is not None:
        out += ["--train-subdir", str(args.train_subdir)]
    if args.test_subdir is not None:
        out += ["--test-subdir", str(args.test_subdir)]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run rule_mask and oracle_mask training+testing sequentially with auto-resume support."
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable")
    parser.add_argument("--only", type=str, default="both", choices=["both", "rule", "oracle"])

    # Common passthrough for both run_train.py scripts.
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--ks", type=str, default=None)
    parser.add_argument("--model-cfg", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-subdir", type=str, default=None)
    parser.add_argument("--test-subdir", type=str, default=None)

    # Optional override for each experiment root.
    parser.add_argument("--rule-exp-root", type=str, default=None)
    parser.add_argument("--oracle-exp-root", type=str, default=None)

    args = parser.parse_args()

    train_root = Path(__file__).resolve().parent
    rule_script = train_root / "rule_mask" / "run_train.py"
    oracle_script = train_root / "oracle_mask" / "run_train.py"

    if not rule_script.exists():
        raise FileNotFoundError(f"rule run script not found: {rule_script}")
    if not oracle_script.exists():
        raise FileNotFoundError(f"oracle run script not found: {oracle_script}")

    common_args = build_common_args(args)

    if args.only in ("both", "rule"):
        cmd = [args.python, str(rule_script)] + common_args
        if args.rule_exp_root is not None:
            cmd += ["--exp-root", str(args.rule_exp_root)]
        print("\n[1/2] rule_mask start", flush=True)
        run_cmd(cmd, cwd=train_root)
        print("[1/2] rule_mask done", flush=True)

    if args.only in ("both", "oracle"):
        cmd = [args.python, str(oracle_script)] + common_args
        if args.oracle_exp_root is not None:
            cmd += ["--exp-root", str(args.oracle_exp_root)]
        idx = "2/2" if args.only == "both" else "1/1"
        print(f"\n[{idx}] oracle_mask start", flush=True)
        run_cmd(cmd, cwd=train_root)
        print(f"[{idx}] oracle_mask done", flush=True)

    print("\n[DONE] All requested runs finished.", flush=True)


if __name__ == "__main__":
    main()
