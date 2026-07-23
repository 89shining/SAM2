#!/usr/bin/env python3
"""Delete erode2/top3/min50/dilate2 prompt files under test patient folders."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PROMPT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)
TARGET_NAMES = (
    "pos_erode2_top3_min50mm2_dilate2.nii.gz",
    "neg_erode2_top3_min50mm2_dilate2.nii.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete only the prompt files generated with erosion radius 2, "
            "Top-K 3, minimum area 50 mm2, and dilation radius 2."
        )
    )
    parser.add_argument("--prompt-root", type=Path, default=DEFAULT_PROMPT_ROOT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag, only print matched files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_dir = args.prompt_root / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

    targets: list[Path] = []
    for patient_dir in sorted(test_dir.glob("p_*")):
        if not patient_dir.is_dir():
            continue
        for name in TARGET_NAMES:
            path = patient_dir / name
            if path.is_file():
                targets.append(path)

    targets = sorted(set(targets))
    if not targets:
        print("No erode2/top3/min50/dilate2 files found. Nothing to delete.")
        return

    action = "DELETE" if args.execute else "PREVIEW"
    for path in targets:
        print(f"[{action}] {path}")

    if not args.execute:
        print(f"\nMatched {len(targets)} files. Re-run with --execute to delete them.")
        return

    for path in targets:
        path.unlink()
    print(
        f"\nDeleted {len(targets)} erode files. All other files were preserved."
    )


if __name__ == "__main__":
    main()
