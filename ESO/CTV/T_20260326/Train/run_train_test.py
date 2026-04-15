#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
import re
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


def build_cuda_alloc_conf(
    max_split_size_mb: int,
    gc_threshold: float,
    expandable_segments: bool,
) -> str:
    conf_items = []
    if int(max_split_size_mb) > 0:
        conf_items.append(f"max_split_size_mb:{int(max_split_size_mb)}")
    if float(gc_threshold) > 0:
        conf_items.append(f"garbage_collection_threshold:{float(gc_threshold)}")
    if expandable_segments:
        conf_items.append("expandable_segments:True")
    return ",".join(conf_items)


def script_supports_arg(script_path: Path, arg_name: str) -> bool:
    try:
        text = script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return arg_name in text


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
    train_root: Path,
    env: dict,
    use_torchrun: bool,
    nproc_per_node: int,
    extra_args: list[str],
) -> None:
    launcher = "python"
    if use_torchrun and nproc_per_node > 1:
        launcher = "torchrun"
        cmd = ["torchrun", f"--nproc_per_node={nproc_per_node}", str(script_path), *extra_args]
    else:
        cmd = [sys.executable, "-u", "-X", "faulthandler", str(script_path), *extra_args]
    safe_step = re.sub(r"[^a-zA-Z0-9._-]+", "_", step_name).strip("_")
    log_dir = train_root / "runner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{safe_step}.log"
    print(f"\n[START] {step_name}")
    print(f"[LAUNCHER] {launcher}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"[LOG] {log_path}")
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
            ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Step failed: {step_name}")
        print(f"[ERROR] Return code: {exc.returncode}")
        print(f"[ERROR] Command: {' '.join(cmd)}")
        print(f"[ERROR] Full child log: {log_path}")
        raise
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
        default=1,
        help="If supported by target train script, call torch.cuda.empty_cache every N batches (0 disables).",
    )
    parser.add_argument(
        "--forward-backbone-per-frame",
        action="store_true",
        default=True,
        help="If supported by target train script, compute image backbone per frame to reduce peak GPU memory.",
    )
    parser.add_argument(
        "--no-forward-backbone-per-frame",
        dest="forward_backbone_per_frame",
        action="store_false",
    )
    parser.add_argument(
        "--cuda-alloc-max-split-mb",
        type=int,
        default=0,
        help="Set max_split_size_mb in PYTORCH_CUDA_ALLOC_CONF (0 disables).",
    )
    parser.add_argument(
        "--cuda-alloc-gc-threshold",
        type=float,
        default=0.0,
        help="Set garbage_collection_threshold in PYTORCH_CUDA_ALLOC_CONF (<=0 disables).",
    )
    parser.add_argument(
        "--cuda-alloc-expandable-segments",
        action="store_true",
        default=False,
        help="Enable expandable_segments in PYTORCH_CUDA_ALLOC_CONF.",
    )
    parser.add_argument(
        "--no-cuda-alloc-expandable-segments",
        dest="cuda_alloc_expandable_segments",
        action="store_false",
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
            ["--resume", "--two-pass-mode", "iterative"],
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
            ["--resume", "--two-pass-mode", "iterative"],
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
            ["--resume", "--two-pass-mode", "iterative"],
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
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    alloc_conf = build_cuda_alloc_conf(
        max_split_size_mb=args.cuda_alloc_max_split_mb,
        gc_threshold=args.cuda_alloc_gc_threshold,
        expandable_segments=args.cuda_alloc_expandable_segments,
    )
    if alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
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
    print(f"[INFO] PYTORCH_CUDA_ALLOC_CONF={env.get('PYTORCH_CUDA_ALLOC_CONF', '<not set>')}")
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
        step_args = list(extra_args)
        if is_train:
            if args.train_empty_cache_every > 0 and script_supports_arg(script_path, "--empty-cache-every"):
                step_args.extend(["--empty-cache-every", str(args.train_empty_cache_every)])
            if args.forward_backbone_per_frame and script_supports_arg(script_path, "--forward-backbone-per-frame"):
                step_args.append("--forward-backbone-per-frame")
        run_step(
            step_name=step_name,
            script_path=script_path,
            repo_root=repo_root,
            train_root=train_root,
            env=env,
            use_torchrun=is_train,
            nproc_per_node=nproc_per_node,
            extra_args=step_args,
        )
        save_state(state_path, idx)

    print("\n[ALL DONE] All train/test steps completed successfully.")


if __name__ == "__main__":
    main()
