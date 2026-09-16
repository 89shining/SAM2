#!/usr/bin/env python
"""Preflight: compare native and official-multiframe P0 on one fixed episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from stage1_bridge import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    DEFAULT_SPLIT_PATH,
    RectalCTVVolumeDataset,
    build_model,
    make_or_load_splits,
)
from stage2_tracking import (
    OfficialMultiFrameBidirectionalState,
    NativeBidirectionalState,
    hard_prediction,
    stacked_logits,
    stacked_native_logits,
)
from training.utils.data_utils import collate_fn


STAGE1_RESULTS = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/"
    "Stage1-mask/TrainResults"
)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, required=True)
    parser.add_argument("--patient-id", type=int, default=9)
    parser.add_argument("--placement", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "train")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument(
        "--validation-plan",
        type=Path,
        default=STAGE1_RESULTS / "validation_prompt_plan.json",
    )
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _model_args(cli):
    return SimpleNamespace(
        model_cfg=cli.model_cfg,
        init_ckpt=cli.init_ckpt,
        input_size=512,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        image_encoder_activation_checkpointing=True,
        fold=cli.fold,
        stage1_ckpt=cli.stage1_ckpt,
    )


def _dice(hard: torch.Tensor, target: torch.Tensor) -> float:
    hard, target = hard.bool(), target.bool()
    denominator = hard.sum() + target.sum()
    if int(denominator) == 0:
        return 1.0
    return float((2 * (hard & target).sum().float() / denominator.float()).item())


def _max_probability_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((torch.sigmoid(left.float()) - torch.sigmoid(right.float())).abs().max())


@torch.no_grad()
def main():
    cli = _args()
    device = torch.device(cli.device)
    patient_dirs = sorted(
        path for path in cli.data_root.iterdir()
        if path.is_dir() and path.name.startswith("p_")
    )
    fold = next(
        item for item in make_or_load_splits(patient_dirs, 5, 20260909, cli.split_path)
        if int(item["fold"]) == cli.fold
    )
    patient_dir = next(
        path for path in fold["val"] if int(path.name.rsplit("_", 1)[1]) == cli.patient_id
    )
    plan = json.loads(cli.validation_plan.read_text(encoding="utf-8"))["folds"][str(cli.fold)]
    placement = plan[str(cli.patient_id)]["placements"]["3"][cli.placement]
    prompts = [int(frame) for frame in placement["prompt_frame_ids"]]
    batch = next(iter(DataLoader(
        RectalCTVVolumeDataset([patient_dir]),
        batch_size=1,
        collate_fn=lambda items: collate_fn(items, dict_key="sam2_p0_equivalence"),
    )))
    batch = batch.to(device)
    model, _ = build_model(cli.model_cfg, cli.init_ckpt, device, _model_args(cli))
    stage1 = torch.load(str(cli.stage1_ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(stage1["model"], strict=True)
    model.eval()

    backbone = model.forward_image(batch.flat_img_batch)
    native = NativeBidirectionalState(model, batch, prompts, backbone)
    adapted = OfficialMultiFrameBidirectionalState(model, batch, prompts, backbone)
    native_outputs, adapted_outputs = native.outputs(), adapted.outputs()
    native_logits = stacked_native_logits(native_outputs)[:, 0]
    adapted_logits = stacked_native_logits(adapted_outputs)[:, 0]
    native_raw = stacked_logits(native_outputs)[:, 0]
    adapted_raw = stacked_logits(adapted_outputs)[:, 0]
    native_hard, adapted_hard = hard_prediction(native_outputs), hard_prediction(adapted_outputs)
    target = batch.masks[:, 0]

    report = {
        "fold": cli.fold,
        "patient_id": cli.patient_id,
        "placement": cli.placement,
        "prompt_frames": prompts,
        "native": {"whole_volume_dice": _dice(native_hard, target)},
        "official_multiframe": {"whole_volume_dice": _dice(adapted_hard, target)},
        "comparison": {
            "hard_voxel_disagreement_fraction": float((native_hard != adapted_hard).float().mean()),
            "native_probability_max_abs_difference": _max_probability_difference(native_logits, adapted_logits),
            "raw_decoder_probability_max_abs_difference": _max_probability_difference(native_raw, adapted_raw),
            "forward_native_probability_max_abs_difference": _max_probability_difference(
                stacked_native_logits(native.forward.outputs())[:, 0],
                stacked_native_logits(adapted.forward.outputs())[:, 0],
            ),
            "reverse_native_probability_max_abs_difference": _max_probability_difference(
                stacked_native_logits(native.reverse.outputs())[:, 0],
                stacked_native_logits(adapted.reverse.outputs())[:, 0],
            ),
        },
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
