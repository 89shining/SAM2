#!/usr/bin/env python3
"""GPU-aware validation probability-cache runner for fixed best checkpoints.

Run only after all 20 training directories contain DONE.  Each job runs exactly
one fixed best.pth with its matched disk radius at deterministic 50% prompts.
No gate is supplied here: gate selection is intentionally offline and later.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


CODE = Path("/home/wusi/SAM2/MyTrain/MyCodes/Rectal/R-20260916-s.pth")
RUN = Path("/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260916-s.pth")
DATA = RUN / "Prompt_mask"
PYTHON = "/home/wusi/miniconda3/envs/sam2/bin/python"
RADII = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
TASKS = [(radius, branch) for radius in RADII for branch in ("pos", "neg")]
MIN_FREE_MIB, MAX_UTILIZATION, MAX_PARALLEL, POLL_SECONDS = 24000, 20, 4, 30


def gpu_state() -> list[tuple[int, int, int]]:
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    return [tuple(map(int, line.split(","))) for line in text.strip().splitlines()]


def command(radius: int, branch: str) -> list[str]:
    checkpoint = RUN / "checkpoints" / branch / f"r{radius:02d}mm" / "checkpoints" / "best.pth"
    split = RUN / "checkpoints" / branch / f"r{radius:02d}mm" / "split.json"
    return [PYTHON, str(CODE / "infer_radius_grid.py"), "--kind", branch,
            "--data-root", str(DATA), "--split-json", str(split), "--subset", "validation",
            "--checkpoint", str(checkpoint), "--output-root",
            str(RUN / "probability_cache" / "validation" / branch / f"r{radius:02d}mm"),
            "--test-radii", str(radius), "--input-size", "512", "--prompt-fraction", "0.50", "--device", "cuda"]


def main() -> None:
    missing = [task for task in TASKS if not (RUN / "checkpoints" / task[1] / f"r{task[0]:02d}mm" / "DONE").is_file()]
    if missing:
        raise RuntimeError(f"Training is incomplete; refusing validation inference: {missing}")
    pending = [task for task in TASKS if not (RUN / "probability_cache" / "validation" / task[1] / f"r{task[0]:02d}mm" / "DONE").is_file()]
    active: dict[int, tuple[subprocess.Popen, object, int, str]] = {}
    while pending or active:
        for gpu, (proc, handle, radius, branch) in list(active.items()):
            if proc.poll() is None:
                continue
            handle.close(); del active[gpu]
            if proc.returncode:
                raise RuntimeError(f"Inference failed: {branch} r={radius} on GPU {gpu}")
            (RUN / "probability_cache" / "validation" / branch / f"r{radius:02d}mm" / "DONE").touch()
        ready = [state for state in gpu_state() if state[0] not in active and state[1] >= MIN_FREE_MIB and state[2] <= MAX_UTILIZATION]
        for gpu, _, _ in sorted(ready, key=lambda item: item[1], reverse=True)[:MAX_PARALLEL-len(active)]:
            if not pending:
                break
            radius, branch = pending.pop(0)
            log = (RUN / "logs" / f"infer_validation_{branch}_r{radius:02d}mm.log").open("a")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            active[gpu] = (subprocess.Popen(command(radius, branch), stdout=log, stderr=subprocess.STDOUT, env=env), log, radius, branch)
        if pending or active:
            time.sleep(POLL_SECONDS)
    (RUN / "VALIDATION_PROBABILITIES_COMPLETE").touch()


if __name__ == "__main__":
    main()
