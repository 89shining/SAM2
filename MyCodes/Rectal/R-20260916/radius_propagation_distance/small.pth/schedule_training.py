#!/usr/bin/env python3
"""Dynamically train the one-fold 2:2:20 mm POS/NEG radius grid."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


CODE = Path("/home/wusi/SAM2/MyTrain/MyCodes/Rectal/R-20260916-s.pth")
RUN = Path("/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260916-s.pth")
DATA = RUN / "Prompt_mask"
SOURCE_SPLIT = Path("/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/TrainResults/POS/split.json")
PYTHON = "/home/wusi/miniconda3/envs/sam2/bin/python"
RADII = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
EXPECTED_TASKS = [(radius, branch) for radius in RADII for branch in ("pos", "neg")]
TASKS_TO_LAUNCH = list(EXPECTED_TASKS)
MIN_FREE_MIB = 16000
MAX_UTILIZATION = 100
# Keep one of the currently eligible A100 GPUs available for other work.
MAX_PARALLEL = 4
MAX_RETRIES = 3
POLL_SECONDS = 30


def gpu_state() -> list[tuple[int, int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = []
    for line in output.strip().splitlines():
        index, free, utilization = (int(part.strip()) for part in line.split(","))
        result.append((index, free, utilization))
    return result


def command(branch: str, radius: int, output: Path) -> list[str]:
    script = CODE / f"fullmask_{branch}.py"
    prompt_flag = "--pos-prompt-name" if branch == "pos" else "--neg-prompt-name"
    prompt_name = f"{branch}_prompt_disk{radius}mm.nii.gz"
    result = [
        PYTHON,
        str(script),
        "--data-root", str(DATA),
        "--output-root", str(output),
        "--test-results-dir", str(output / "unused_test"),
        prompt_flag, prompt_name,
        "--input-size", "512",
        "--max-epochs", "60",
        "--patience", "10",
        "--warmup-epochs", "5",
        "--early-stop-start-epoch", "5",
        "--train-prompt-keep-min", "0.30",
        "--train-prompt-keep-max", "0.70",
        "--eval-prompt-keep", "0.50",
        "--train-temporal-window", "9",
        "--skip-test",
    ]
    if branch == "pos":
        result.extend(("--debug-epochs", "0"))
    return result


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    prerequisites = (RUN / "PROMPTS_READY", RUN / "PREFLIGHT_PASS", RUN / "PROMPT_SLICE_CONSISTENCY_PASS")
    while not all(path.is_file() for path in prerequisites):
        print(f"[{time.strftime('%F %T')}] waiting for prerequisites: {prerequisites}", flush=True)
        time.sleep(POLL_SECONDS)

    tasks = list(TASKS_TO_LAUNCH)
    active: dict[int, tuple[subprocess.Popen, object, int, str, int]] = {}
    attempts: dict[tuple[int, str], int] = {}
    while tasks or active:
        for gpu, (process, log_handle, radius, branch, _) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle.close()
            del active[gpu]
            output = RUN / "checkpoints" / branch / f"r{radius:02d}mm"
            if returncode == 0:
                (output / "DONE").touch()
                print(f"[{time.strftime('%F %T')}] DONE {branch} r={radius} on GPU {gpu}", flush=True)
            else:
                task = (radius, branch)
                attempts[task] = attempts.get(task, 0) + 1
                if attempts[task] >= MAX_RETRIES:
                    (RUN / "FAILED_PERMANENTLY").write_text(
                        f"{branch} r={radius} failed {attempts[task]} times; see logs\\n"
                    )
                    raise RuntimeError(f"{branch} r={radius} exceeded MAX_RETRIES={MAX_RETRIES}")
                print(f"[{time.strftime('%F %T')}] FAILED {branch} r={radius}; retry {attempts[task]}/{MAX_RETRIES}", flush=True)
                tasks.append(task)

        tasks = [
            task for task in tasks
            if not (RUN / "checkpoints" / task[1] / f"r{task[0]:02d}mm" / "DONE").is_file()
        ]
        if tasks and len(active) < MAX_PARALLEL:
            candidates = [
                state for state in gpu_state()
                if state[0] in (2, 3, 4, 5) and state[0] not in active
                and state[1] >= MIN_FREE_MIB
                and state[2] <= MAX_UTILIZATION
            ]
            candidates.sort(key=lambda state: state[1], reverse=True)
            for gpu, free, utilization in candidates[: MAX_PARALLEL - len(active)]:
                if not tasks:
                    break
                radius, branch = tasks.pop(0)
                output = RUN / "checkpoints" / branch / f"r{radius:02d}mm"
                output.mkdir(parents=True, exist_ok=True)
                split_source = SOURCE_SPLIT
                shutil.copy2(split_source, output / "split.json")
                log_path = RUN / "logs" / f"train_{branch}_r{radius:02d}mm.log"
                log_handle = log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command(branch, radius, output),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                active[gpu] = (process, log_handle, radius, branch, free)
                print(
                    f"[{time.strftime('%F %T')}] START {branch} r={radius} on GPU {gpu} "
                    f"(free={free} MiB, util={utilization}%)",
                    flush=True,
                )
        if tasks or active:
            time.sleep(POLL_SECONDS)

    while True:
        missing = [
            (radius, branch) for radius, branch in EXPECTED_TASKS
            if not (RUN / "checkpoints" / branch / f"r{radius:02d}mm" / "DONE").is_file()
        ]
        if not missing:
            break
        print(f"[{time.strftime('%F %T')}] waiting for pre-existing tasks: {missing}", flush=True)
        time.sleep(POLL_SECONDS)
    (RUN / "TRAINING_COMPLETE_A100").touch()
    print(f"[{time.strftime('%F %T')}] all A100 whitelist models complete", flush=True)


if __name__ == "__main__":
    main()
