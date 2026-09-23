#!/usr/bin/env python3
"""Bidirectional inference for fixed skeleton thicknesses."""
from __future__ import annotations

import argparse
import json
import re
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from skeleton_prompt_ops import build_skeleton_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("pos", "neg"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--subset", choices=("validation", "test"), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thicknesses-mm", type=float, nargs="+", required=True)
    parser.add_argument("--segment-fraction", type=float, default=0.50)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--prompt-fraction", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def patient_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def build_branch(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.kind == "pos":
        import fullmask_pos as branch
        model_args = Namespace(
            model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
            input_size=args.input_size,
            init_checkpoint=Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_small.pt"),
            lora_r=4, lora_alpha=16, lora_dropout=0.1,
        )
    else:
        import fullmask_neg as branch
        model_args = Namespace(
            model_config="configs/sam2.1/sam2.1_hiera_s.yaml",
            input_size=args.input_size,
            init_checkpoint=Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_small.pt"),
            lora_r=4, lora_alpha=16, lora_dropout=0.1,
        )
    model, _ = branch.build_model(model_args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.train(False)
    return branch, model


def cases(args: argparse.Namespace) -> list[Path]:
    if args.subset == "validation":
        names = json.loads(args.split_json.read_text())["validation"]
        root = args.data_root / "train"
    else:
        root = args.data_root / "test"
        names = [path.name for path in root.glob("p_*")]
    return sorted((root / name for name in names), key=patient_number)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    branch, model = build_branch(args, device)
    if not 0 < args.segment_fraction <= 1:
        raise ValueError("--segment-fraction must be in (0, 1]")
    for thickness in args.thicknesses_mm:
        thickness_tag = f"t{int(thickness):02d}mm"
        prompt_name = f"{args.kind}_skeleton.nii.gz"
        output_dir = args.output_root / thickness_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_selection: dict[str, np.ndarray] = {}
        for case in cases(args):
            output = output_dir / f"{case.name}.npz"
            if output.is_file():
                continue
            if args.kind == "pos":
                dataset = branch.PositiveCorrectionDataset(
                    [case], prompt_name, args.input_size, 40.0, 400.0,
                    thickness_mm=thickness, segment_keep_min=args.segment_fraction,
                    segment_mode="center", training_prompt=False,
                )
            else:
                dataset = branch.NegativeCorrectionDataset(
                    [case], prompt_name, args.input_size, 40.0, 400.0,
                    thickness_mm=thickness, segment_keep_min=args.segment_fraction,
                    segment_mode="center", training_prompt=False,
                )
            cpu_batch = dataset[0]
            ref = sitk.ReadImage(str(case / "image.nii.gz"))
            skeleton = sitk.GetArrayFromImage(sitk.ReadImage(str(case / prompt_name))) > 0
            original = build_skeleton_prompt(skeleton, ref.GetSpacing(), thickness, args.segment_fraction, "center")
            if args.prompt_fraction == 0 or not original.any():
                # Explicit 0% condition: no prompt is sent to SAM2 and fusion is
                # exactly the original nnU-Net baseline downstream.
                probability = np.zeros(original.shape, dtype=np.float16)
                selected = np.empty(0, dtype=np.int16)
                valid = np.zeros(original.shape[0], dtype=bool)
            else:
                batch = cpu_batch.to(device, non_blocking=True)
                selected_list = (
                    branch.positive_prompt_frames(batch, args.prompt_fraction)
                    if args.kind == "pos"
                    else branch.prompt_frames(batch, args.prompt_fraction)
                )
                selected = np.asarray(selected_list, dtype=np.int16)
                with torch.no_grad(), torch.cuda.amp.autocast(
                    enabled=device.type == "cuda", dtype=torch.bfloat16
                ):
                    logits = branch.bidirectional_logits(
                        model, batch, prompt_keep_fraction=args.prompt_fraction
                    )
                    resized = F.interpolate(
                        torch.sigmoid(logits.float()),
                        size=batch.original_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                probability = resized[:, 0].cpu().numpy().astype(np.float16)
                valid = np.ones(original.shape[0], dtype=bool)
                del batch, logits, resized
                torch.cuda.empty_cache()
            previous = reference_selection.setdefault(case.name, selected)
            if not np.array_equal(previous, selected):
                raise RuntimeError(
                    f"Selected prompt slices changed across test radii: {case.name}"
                )
            np.savez_compressed(
                output,
                probability=probability,
                selected_prompt_slices=selected,
                valid_slices=valid,
                prompt_mask=np.where(np.isin(np.arange(original.shape[0])[:, None, None], selected), original, False).astype(np.uint8),
                prompt_fraction=np.float32(args.prompt_fraction),
                skeleton_thickness_mm=np.float32(thickness),
                segment_fraction=np.float32(args.segment_fraction),
            )
            print(args.kind, thickness_tag, case.name, selected.tolist(), flush=True)
    (args.output_root / "DONE").touch()


if __name__ == "__main__":
    main()
