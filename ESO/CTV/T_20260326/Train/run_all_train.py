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


def resolve_script(train_root: Path, rel_path: str) -> Path:
    script = train_root / rel_path
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    return script


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run all training pipelines in fixed order with unified CUDA setting:\n"
            "1) bad_hd95/mask_prompt_3/one_epoch/run_train.py\n"
            "2) bad_hd95/mask_prompt_3/two_epoch/run_train.py\n"
            "3) oracle_mask/mask_prompt_3/two_epoch/run_train.py\n"
            "All sub-pipelines use their own auto-resume logic."
        )
    )
    parser.add_argument("--gpu", type=str, default="5", help="CUDA_VISIBLE_DEVICES value for all stages")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun nproc per node for all stages")
    parser.add_argument("--torchrun", type=str, default="torchrun", help="torchrun executable")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable")
    parser.add_argument("--only", type=str, default="", help="Optional comma list: bad_one,bad_two,oracle_two")
    args = parser.parse_args()

    train_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    all_stages = [
        (
            "bad_one",
            "Bad_hd95 one_epoch",
            resolve_script(train_root, "Bad_hd95/mask_prompt_3/one_epoch/run_train.py"),
        ),
        (
            "bad_two",
            "Bad_hd95 two_epoch_fix",
            resolve_script(train_root, "Bad_hd95/mask_prompt_3/two_epoch_fix/run_train.py"),
        ),
        (
            "oracle_two",
            "Oracle two_epoch",
            resolve_script(train_root, "oracle_mask/mask_prompt_3/two_epoch/run_train.py"),
        ),
    ]

    selected = None
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [stage for stage in all_stages if stage[0] in wanted]
        if len(selected) == 0:
            raise ValueError(f"--only matched no stages: {args.only}")
    else:
        selected = all_stages

    print(f"[INFO] Train root: {train_root}", flush=True)
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print("[INFO] Resume policy: enabled by each sub run_train.py", flush=True)
    print("[INFO] Execution order:", flush=True)
    for idx, (_, title, script) in enumerate(selected, start=1):
        print(f"  {idx}. {title} -> {script}", flush=True)

    for idx, (_, title, script) in enumerate(selected, start=1):
        print(f"\n[{idx}/{len(selected)}] Start: {title}", flush=True)
        cmd = [
            args.python,
            str(script),
            "--gpu",
            args.gpu,
            "--nproc-per-node",
            str(args.nproc_per_node),
            "--torchrun",
            args.torchrun,
            "--python",
            args.python,
        ]
        run_cmd(cmd, env=env, cwd=str(script.parent))
        print(f"[{idx}/{len(selected)}] Done: {title}", flush=True)

    print("\n[DONE] All selected training stages finished.", flush=True)


if __name__ == "__main__":
    main()
