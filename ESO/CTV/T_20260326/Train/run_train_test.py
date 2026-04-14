#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nohup bash -lc '
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python -u run_train_test.py
' > run_train_test.log 2>&1 &
echo $!

tail -f run_train_test.log

"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    # Support both .../SAM2/ESO/... and possible relocated script paths.
    for p in [start] + list(start.parents):
        if p.name == "SAM2" and (p / "ESO").is_dir():
            return p
    raise RuntimeError(f"Cannot locate SAM2 repo root from: {start}")


def parse_visible_cuda_count(env: dict) -> int:
    cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return 1
    parts = [x.strip() for x in cvd.split(",") if x.strip()]
    if not parts:
        return 1
    if len(parts) == 1 and parts[0] == "-1":
        return 1
    return max(1, len(parts))


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"last_completed_step": -1}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_completed_step": -1}


def save_state(state_path: Path, step_idx: int) -> None:
    state_path.write_text(
        json.dumps({"last_completed_step": step_idx}, indent=2),
        encoding="utf-8",
    )


def run_step(
    step_name: str,
    script_path: Path,
    repo_root: Path,
    env: dict,
    use_torchrun: bool,
    nproc_per_node: int,
    extra_args: list[str],
) -> None:
    if use_torchrun:
        cmd = ["torchrun", f"--nproc_per_node={nproc_per_node}", str(script_path), *extra_args]
    else:
        cmd = [sys.executable, str(script_path), *extra_args]
    print(f"\n[START] {step_name}")
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    print(f"[DONE] {step_name}")


def main() -> None:
    parser = argparse.ArgumentParser("Run hd95/oracle/middle iterative train+test in order with resume support")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=0,
        help="Torchrun worker count for TRAIN steps. 0 means auto from CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to runner state json. Default: <this_dir>/run_train_test_state.json",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore previous state and run from beginning.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=None,
        help="Force start from step index (1-based). Overrides saved state if provided.",
    )
    parser.add_argument(
        "--train-empty-cache-every",
        type=int,
        default=0,
        help="Pass --empty-cache-every to each TRAIN script (0 disables).",
    )
    parser.add_argument(
        "--cuda-alloc-max-split-mb",
        type=int,
        default=128,
        help="Pass --cuda-alloc-max-split-mb to each TRAIN script (0 disables).",
    )
    parser.add_argument(
        "--cuda-alloc-gc-threshold",
        type=float,
        default=0.8,
        help="Pass --cuda-alloc-gc-threshold to each TRAIN script (<=0 disables).",
    )
    parser.add_argument(
        "--no-cuda-alloc-expandable-segments",
        action="store_true",
        help="Pass --no-cuda-alloc-expandable-segments to each TRAIN script.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)

    # Compatible with script placed in current Train dir or nested under one more level.
    if (script_dir / "Bad_hd95").is_dir() and (script_dir / "oracle_mask").is_dir() and (script_dir / "rule_mask").is_dir():
        train_root = script_dir
    elif (script_dir / "two_epoch" / "Bad_hd95").is_dir():
        train_root = script_dir / "two_epoch"
    else:
        train_root = repo_root / "ESO" / "CTV" / "T_20260326" / "Train"

    state_path = args.state_file if args.state_file is not None else (train_root / "run_train_test_state.json")

    train_common_args = [
        "--resume",
        "--two-pass-mode",
        "iterative",
        "--forward-backbone-per-frame",
        "--empty-cache-every",
        str(args.train_empty_cache_every),
        "--cuda-alloc-max-split-mb",
        str(args.cuda_alloc_max_split_mb),
        "--cuda-alloc-gc-threshold",
        str(args.cuda_alloc_gc_threshold),
    ]
    if args.no_cuda_alloc_expandable_segments:
        train_common_args.append("--no-cuda-alloc-expandable-segments")

    # Order required by user: hd95 -> oracle -> middle(rule middle)
    steps = [
        (
            "1) hd95 TRAIN",
            train_root
            / "Bad_hd95"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_online_hd95_middle_cv.py",
            True,
            list(train_common_args),
        ),
        (
            "1) hd95 TEST",
            train_root
            / "Bad_hd95"
            / "mask_prompt_3"
            / "two_epoch"
            / "test_upper_lower_online_hd95_middle.py",
            False,
            ["--two-pass-mode", "iterative"],
        ),
        (
            "2) oracle TRAIN",
            train_root
            / "oracle_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_middle_cv_two_epoch.py",
            True,
            list(train_common_args),
        ),
        (
            "2) oracle TEST",
            train_root
            / "oracle_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "test_upper_lower_middle.py",
            False,
            ["--two-pass-mode", "iterative"],
        ),
        (
            "3) middle(rule) TRAIN",
            train_root
            / "rule_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "train_upper_lower_middle_rule_cv.py",
            True,
            list(train_common_args),
        ),
        (
            "3) middle(rule) TEST",
            train_root
            / "rule_mask"
            / "mask_prompt_3"
            / "two_epoch"
            / "test_upper_lower_middle_rule.py",
            False,
            ["--two-pass-mode", "iterative"],
        ),
    ]

    missing = [str(p) for _, p, _, _ in steps if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing script(s):\n" + "\n".join(missing))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.nproc_per_node > 0:
        nproc_per_node = args.nproc_per_node
    else:
        nproc_per_node = parse_visible_cuda_count(env)

    state = {"last_completed_step": -1} if args.restart else load_state(state_path)
    if args.from_step is not None:
        # step is 1-based for user friendliness
        state["last_completed_step"] = max(-1, int(args.from_step) - 2)

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Train root: {train_root}")
    print(f"[INFO] Runner state file: {state_path}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"[INFO] torchrun --nproc_per_node={nproc_per_node} for TRAIN steps (if TRAIN)")
    print("[INFO] Running in strict order:")
    print("[INFO]   1) hd95 train -> test")
    print("[INFO]   2) oracle train -> test")
    print("[INFO]   3) middle(rule) train -> test")
    print(f"[INFO] Resume from step index: {state.get('last_completed_step', -1) + 1}")

    last_done = int(state.get("last_completed_step", -1))
    for idx, (step_name, script_path, is_train, extra_args) in enumerate(steps):
        if idx <= last_done:
            print(f"[SKIP] {step_name} (already completed)")
            continue
        run_step(
            step_name=step_name,
            script_path=script_path,
            repo_root=repo_root,
            env=env,
            use_torchrun=is_train,
            nproc_per_node=nproc_per_node,
            extra_args=extra_args,
        )
        save_state(state_path, idx)

    print("\n[ALL DONE] All train/test steps completed successfully.")


if __name__ == "__main__":
    main()
