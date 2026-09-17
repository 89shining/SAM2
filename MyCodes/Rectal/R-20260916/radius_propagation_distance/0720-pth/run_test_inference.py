#!/usr/bin/env python3
"""Run fixed-checkpoint, deterministic test probability inference.

This runner deliberately does not inspect metrics or gates.  It produces the
probability cache needed by the downstream test evaluator, one cache for each
already-trained (branch, radius) model.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


RADII = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
GPU_IDS = (2, 3, 4, 5)  # GPU 6 remains reserved for the user.
MIN_FREE_MIB = 16_000
POLL_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def gpu_state() -> list[tuple[int, int]]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [tuple(map(int, line.split(","))) for line in text.strip().splitlines()]


def main() -> None:
    args = parse_args()
    run = args.run_root.resolve()
    code = Path(__file__).resolve().parent
    python = os.environ.get("SAM2_PYTHON", "/home/wusi/miniconda3/envs/sam2/bin/python")
    data = run / "Prompt_mask"
    tasks = [(radius, branch) for radius in RADII for branch in ("pos", "neg")]
    missing = [
        task for task in tasks
        if not (run / "checkpoints" / task[1] / f"r{task[0]:02d}mm" / "DONE").is_file()
    ]
    if missing:
        raise RuntimeError(f"Training is incomplete; refusing test inference: {missing}")

    pending = [
        task for task in tasks
        if not (run / "probability_cache" / "test" / task[1] / f"r{task[0]:02d}mm" / "DONE").is_file()
    ]
    active: dict[int, tuple[subprocess.Popen, object, int, str]] = {}
    (run / "logs").mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu, (process, handle, radius, branch) in list(active.items()):
            if process.poll() is None:
                continue
            handle.close()
            del active[gpu]
            if process.returncode:
                raise RuntimeError(f"Test inference failed: {branch} r={radius} on GPU {gpu}")
            (run / "probability_cache" / "test" / branch / f"r{radius:02d}mm" / "DONE").touch()
            print(f"[{time.strftime('%F %T')}] DONE {branch} r={radius} on GPU {gpu}", flush=True)

        ready = [
            (gpu, free) for gpu, free in gpu_state()
            if gpu in GPU_IDS and gpu not in active and free >= MIN_FREE_MIB
        ]
        for gpu, free in sorted(ready, key=lambda item: item[1], reverse=True)[: len(GPU_IDS) - len(active)]:
            if not pending:
                break
            radius, branch = pending.pop(0)
            checkpoint_root = run / "checkpoints" / branch / f"r{radius:02d}mm"
            output = run / "probability_cache" / "test" / branch / f"r{radius:02d}mm"
            output.mkdir(parents=True, exist_ok=True)
            prompt_flag = "--pos-prompt-name" if branch == "pos" else "--neg-prompt-name"
            command = [
                python, str(code / "infer_radius_grid.py"), "--kind", branch,
                "--data-root", str(data), "--split-json", str(checkpoint_root / "split.json"),
                "--subset", "test", "--checkpoint", str(checkpoint_root / "checkpoints" / "best.pth"),
                "--output-root", str(output), "--test-radii", str(radius),
                "--input-size", "512", "--prompt-fraction", "0.50", "--device", "cuda",
            ]
            # The prompt filename is inferred by infer_radius_grid from kind/radius.
            del prompt_flag
            log = (run / "logs" / f"infer_test_{branch}_r{radius:02d}mm.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            active[gpu] = (
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment),
                log, radius, branch,
            )
            print(f"[{time.strftime('%F %T')}] START {branch} r={radius} on GPU {gpu} (free={free} MiB)", flush=True)
        if pending or active:
            time.sleep(POLL_SECONDS)
    (run / "TEST_PROBABILITIES_COMPLETE").touch()


if __name__ == "__main__":
    main()
