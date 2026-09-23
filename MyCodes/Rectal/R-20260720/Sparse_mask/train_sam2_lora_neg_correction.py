#!/usr/bin/env python3
"""Negative-only SAM2-LoRA training, validation, and bidirectional testing.

Inputs
------
Image encoder:
    image.nii.gz and nnunet.nii.gz through a shared SAM2 image encoder.
Prompt encoder:
    neg_erode2_top3_min50mm2_dilate2.nii.gz only.
Training target:
    nnunet AND NOT neg_raw.
True evaluation target:
    CTV.nii.gz.

This stage only learns to remove nnU-Net false positives. It deliberately does
not add nnU-Net false negatives. Patients under train/ are split once into
90% training and 10% validation. The best checkpoint is automatically tested
bidirectionally on test/, and predictions are saved as CTV_XXX.nii.gz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset


CURRENT_DIR = Path(__file__).resolve().parent


def find_project_root() -> Path:
    candidates = [CURRENT_DIR] + list(CURRENT_DIR.parents)
    configured = os.environ.get("SAM2_PROJECT_ROOT", "").strip()
    if configured:
        candidates.insert(0, Path(configured).resolve())
    for candidate in candidates:
        if (candidate / "sam2").is_dir() and (candidate / "training").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Cannot find SAM2 root; set SAM2_PROJECT_ROOT.")


PROJECT_ROOT = find_project_root()

from sam2.modeling.lora import (  # noqa: E402
    LoRAConfig,
    apply_lora,
    apply_qv_lora_to_fused_qkv,
)
from training.model.sam2 import SAM2Train  # noqa: E402


DATA_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)
OUTPUT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/"
    "NegativeCorrection/TrainResults"
)
TEST_RESULTS = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/"
    "NegativeCorrection/TestResults"
)
INIT_CHECKPOINT = Path(
    "/home/wusi/SAM2/checkpoints/sam2.1_hiera_small.pt"
)
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
NEG_PROMPT_NAME = "neg_erode2_top3_min50mm2_dilate2.nii.gz"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_number(patient: Path) -> int:
    match = re.search(r"(\d+)$", patient.name)
    if match is None:
        raise ValueError(f"Cannot parse patient number: {patient}")
    return int(match.group(1))


def list_patients(root: Path, prompt_name: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    patients = sorted(
        (path for path in root.glob("p_*") if path.is_dir()),
        key=patient_number,
    )
    if not patients:
        raise RuntimeError(f"No p_* folders found: {root}")
    required = (
        "image.nii.gz",
        "CTV.nii.gz",
        "nnunet.nii.gz",
        "neg_raw.nii.gz",
        prompt_name,
    )
    missing = [
        str(patient / name)
        for patient in patients
        for name in required
        if not (patient / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing[:20])
        )
    return patients


def make_split(
    patients: Sequence[Path],
    fraction: float,
    seed: int,
    path: Path,
) -> tuple[list[Path], list[Path]]:
    patient_map = {patient.name: patient for patient in patients}
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return (
            [patient_map[name] for name in saved["train"]],
            [patient_map[name] for name in saved["validation"]],
        )
    if len(patients) < 2:
        raise RuntimeError("At least two patients are required.")
    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = sorted(shuffled[:validation_count], key=patient_number)
    train = sorted(shuffled[validation_count:], key=patient_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "validation_fraction": fraction,
                "train": [patient.name for patient in train],
                "validation": [patient.name for patient in validation],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return train, validation


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-5, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0)
    )


def window_ct(array: np.ndarray, center: float, width: float) -> np.ndarray:
    low, high = center - width / 2.0, center + width / 2.0
    return (
        (np.clip(array.astype(np.float32), low, high) - low)
        / (high - low)
    )


@dataclass
class NegativeBatch:
    images: torch.Tensor
    targets: torch.Tensor
    true_masks: torch.Tensor
    nnunet_masks: torch.Tensor
    negative_errors: torch.Tensor
    negative_prompts: torch.Tensor
    patient_dir: Path
    original_hw: tuple[int, int]

    @property
    def num_frames(self) -> int:
        return int(self.images.shape[0])

    @property
    def flat_img_batch(self) -> torch.Tensor:
        return self.images

    @property
    def flat_obj_to_img_idx(self) -> torch.Tensor:
        return torch.arange(
            self.num_frames, device=self.images.device
        ).unsqueeze(1)

    def to(self, device: torch.device, non_blocking: bool = False):
        return NegativeBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            true_masks=self.true_masks.to(device, non_blocking=non_blocking),
            nnunet_masks=self.nnunet_masks.to(
                device, non_blocking=non_blocking
            ),
            negative_errors=self.negative_errors.to(
                device, non_blocking=non_blocking
            ),
            negative_prompts=self.negative_prompts.to(
                device, non_blocking=non_blocking
            ),
            patient_dir=self.patient_dir,
            original_hw=self.original_hw,
        )

    def pin_memory(self):
        return NegativeBatch(
            images=self.images.pin_memory(),
            targets=self.targets.pin_memory(),
            true_masks=self.true_masks.pin_memory(),
            nnunet_masks=self.nnunet_masks.pin_memory(),
            negative_errors=self.negative_errors.pin_memory(),
            negative_prompts=self.negative_prompts.pin_memory(),
            patient_dir=self.patient_dir,
            original_hw=self.original_hw,
        )


class NegativeCorrectionDataset(Dataset):
    def __init__(
        self,
        patients: Sequence[Path],
        prompt_name: str,
        input_size: int,
        window_center: float,
        window_width: float,
    ) -> None:
        self.patients = list(patients)
        self.prompt_name = prompt_name
        self.input_size = input_size
        self.window_center = window_center
        self.window_width = window_width

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> NegativeBatch:
        patient = self.patients[index]
        paths = {
            "image": patient / "image.nii.gz",
            "ctv": patient / "CTV.nii.gz",
            "nnunet": patient / "nnunet.nii.gz",
            "neg_raw": patient / "neg_raw.nii.gz",
            "prompt": patient / self.prompt_name,
        }
        images = {key: sitk.ReadImage(str(path)) for key, path in paths.items()}
        reference = images["image"]
        for key, image in images.items():
            if not same_geometry(reference, image):
                raise ValueError(f"Geometry mismatch in {patient}: {key}")
        arrays = {
            key: sitk.GetArrayFromImage(image)
            for key, image in images.items()
        }
        shapes = {key: array.shape for key, array in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Shape mismatch in {patient}: {shapes}")

        ct = window_ct(
            arrays["image"], self.window_center, self.window_width
        )
        nnunet = arrays["nnunet"] > 0
        ctv = arrays["ctv"] > 0
        neg_raw = arrays["neg_raw"] > 0
        neg_prompt = arrays["prompt"] > 0

        # Exact negative-only GT:
        # neg_raw = nnunet & ~CTV
        # target  = nnunet & ~neg_raw = nnunet & CTV
        target = nnunet & ~neg_raw

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array.astype(np.float32)).unsqueeze(1)

        size = (self.input_size, self.input_size)
        ct_tensor = F.interpolate(
            tensor(ct), size=size, mode="bilinear", align_corners=False
        )
        nnunet_tensor = F.interpolate(
            tensor(nnunet), size=size, mode="nearest"
        )
        target_tensor = F.interpolate(
            tensor(target), size=size, mode="nearest"
        )
        ctv_tensor = F.interpolate(tensor(ctv), size=size, mode="nearest")
        neg_raw_tensor = F.interpolate(
            tensor(neg_raw), size=size, mode="nearest"
        )
        prompt_tensor = F.interpolate(
            tensor(neg_prompt), size=size, mode="nearest"
        )

        mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1)
        ct_rgb = (ct_tensor.repeat(1, 3, 1, 1) - mean) / std
        nnunet_rgb = (nnunet_tensor.repeat(1, 3, 1, 1) - mean) / std
        return NegativeBatch(
            images=torch.cat((ct_rgb, nnunet_rgb), dim=1).float(),
            targets=target_tensor.bool(),
            true_masks=ctv_tensor.bool(),
            nnunet_masks=nnunet_tensor.bool(),
            negative_errors=neg_raw_tensor.bool(),
            negative_prompts=prompt_tensor.float(),
            patient_dir=patient,
            original_hw=(ct.shape[1], ct.shape[2]),
        )


def collate_one(items: list[NegativeBatch]) -> NegativeBatch:
    if len(items) != 1:
        raise ValueError("batch_size must be 1")
    return items[0]


class DualStreamSAM2Train(SAM2Train):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial = math.log(0.1 / 0.9)
        self.dual_stream_feature_gates = nn.Parameter(
            torch.full((3,), initial)
        )

    def forward_image(self, images: torch.Tensor) -> dict:
        if images.ndim != 4 or images.shape[1] != 6:
            raise ValueError(f"Expected [N,6,H,W], got {tuple(images.shape)}")
        ct_output = super().forward_image(images[:, :3])
        nnunet_output = super().forward_image(images[:, 3:])
        gates = torch.sigmoid(self.dual_stream_feature_gates)
        ct_fpn = ct_output["backbone_fpn"]
        nnunet_fpn = nnunet_output["backbone_fpn"]
        if len(ct_fpn) != 3 or len(nnunet_fpn) != 3:
            raise RuntimeError("Expected three SAM2 FPN levels.")
        output = dict(ct_output)
        output["backbone_fpn"] = [
            ct + gates[level].to(ct.dtype) * prior
            for level, (ct, prior) in enumerate(zip(ct_fpn, nnunet_fpn))
        ]
        output["vision_pos_enc"] = ct_output["vision_pos_enc"]
        return output


def load_config(name: str) -> dict:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module("sam2", version_base="1.2"):
        config = compose(config_name=name)
    return OmegaConf.to_container(config.model, resolve=True)


def build_model(args, device: torch.device):
    config = load_config(args.model_config)
    config["image_size"] = args.input_size
    config["freeze_image_encoder"] = False
    # Required: neg prompt must pass through Prompt Encoder and Mask Decoder.
    config["use_mask_input_as_output_without_sam"] = False
    config["pred_obj_scores"] = False
    config["fixed_no_obj_ptr"] = False
    config["multimask_output_in_sam"] = False
    config["multimask_output_for_tracking"] = False
    image_encoder = instantiate(config.pop("image_encoder"), _recursive_=True)
    memory_attention = instantiate(
        config.pop("memory_attention"), _recursive_=True
    )
    memory_encoder = instantiate(
        config.pop("memory_encoder"), _recursive_=True
    )
    config.pop("_target_", None)
    model = DualStreamSAM2Train(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        num_frames_to_correct_for_train=1,
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        num_correction_pt_per_frame=0,
        rand_init_cond_frames_for_train=False,
        rand_init_cond_frames_for_eval=False,
        **config,
    )
    checkpoint = torch.load(str(args.init_checkpoint), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)

    for parameter in model.parameters():
        parameter.requires_grad = False
    common = dict(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        freeze_base_model=False,
    )
    image_count = apply_qv_lora_to_fused_qkv(
        model,
        LoRAConfig(
            target_modules=("qkv",),
            target_prefixes=("image_encoder",),
            **common,
        ),
    )
    attention_count = apply_lora(
        model,
        LoRAConfig(
            target_modules=("q_proj", "v_proj"),
            target_prefixes=("memory_attention",),
            **common,
        ),
    )
    encoder_count = apply_lora(
        model,
        LoRAConfig(
            target_modules=("pwconv1", "pwconv2"),
            target_prefixes=("memory_encoder.fuser",),
            **common,
        ),
    )
    for module in (model.sam_prompt_encoder, model.sam_mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.dual_stream_feature_gates.requires_grad = True
    if min(image_count, attention_count, encoder_count) == 0:
        raise RuntimeError(
            f"LoRA installation failed: image={image_count}, "
            f"memory_attention={attention_count}, "
            f"memory_encoder={encoder_count}"
        )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    stats = {
        "image_encoder_lora_layers": image_count,
        "memory_attention_lora_layers": attention_count,
        "memory_encoder_lora_layers": encoder_count,
        "trainable_params": trainable,
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    return model.to(device), stats


def build_optimizer(model, args):
    adapters, modules = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name or name == "dual_stream_feature_gates":
            adapters.append(parameter)
        else:
            modules.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": adapters, "lr": args.adapter_lr},
            {"params": modules, "lr": args.module_lr},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer, args):
    def schedule(base_lr: float):
        minimum = args.min_lr / base_lr

        def factor(epoch: int) -> float:
            if epoch < args.warmup_epochs:
                return (epoch + 1) / max(1, args.warmup_epochs)
            progress = (epoch - args.warmup_epochs) / max(
                1, args.max_epochs - args.warmup_epochs
            )
            return max(minimum, 0.5 * (1 + math.cos(math.pi * progress)))

        return factor

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [schedule(group["lr"]) for group in optimizer.param_groups],
    )


def prompt_frames(batch: NegativeBatch) -> list[int]:
    nonempty = batch.negative_prompts.flatten(1).any(dim=1)
    frames = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
    return [int(frame) for frame in frames] if frames else [0]


def track_direction(
    model: DualStreamSAM2Train,
    backbone: dict,
    batch: NegativeBatch,
    prompts: Sequence[int],
    reverse: bool,
) -> list[dict]:
    _, features, positions, sizes = model._prepare_backbone_features(backbone)
    prompts = sorted(set(prompts), reverse=reverse)
    prompt_set = set(prompts)
    remaining = [
        frame for frame in range(batch.num_frames) if frame not in prompt_set
    ]
    remaining.sort(reverse=reverse)
    outputs, all_outputs = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }, {}
    for frame in prompts + remaining:
        image_ids = batch.flat_obj_to_img_idx[frame]
        current = model.track_step(
            frame_idx=frame,
            is_init_cond_frame=frame in prompt_set,
            current_vision_feats=[
                feature[:, image_ids] for feature in features
            ],
            current_vision_pos_embeds=[
                position[:, image_ids] for position in positions
            ],
            feat_sizes=sizes,
            point_inputs=None,
            mask_inputs=(
                batch.negative_prompts[frame].unsqueeze(1)
                if frame in prompt_set
                else None
            ),
            gt_masks=batch.targets[frame].unsqueeze(1),
            frames_to_add_correction_pt=[],
            output_dict=outputs,
            num_frames=batch.num_frames,
            track_in_reverse=reverse,
            prev_sam_mask_logits=None,
        )
        key = (
            "cond_frame_outputs"
            if frame in prompt_set
            else "non_cond_frame_outputs"
        )
        outputs[key][frame] = current
        all_outputs[frame] = current
    return [all_outputs[frame] for frame in range(batch.num_frames)]


def bidirectional_logits(
    model: DualStreamSAM2Train, batch: NegativeBatch
) -> torch.Tensor:
    backbone = model.forward_image(batch.flat_img_batch)
    prompts = prompt_frames(batch)
    forward = track_direction(model, backbone, batch, prompts, False)
    backward = track_direction(model, backbone, batch, prompts, True)
    forward_logits = torch.stack(
        [output["pred_masks_high_res"][:, 0] for output in forward]
    )
    backward_logits = torch.stack(
        [output["pred_masks_high_res"][:, 0] for output in backward]
    )
    probability = 0.5 * (
        torch.sigmoid(forward_logits) + torch.sigmoid(backward_logits)
    )
    return torch.logit(probability.clamp(1e-4, 1 - 1e-4))


class DiceBCELoss(nn.Module):
    def forward(self, logits, targets):
        targets = targets.float()
        foreground = targets.sum()
        background = targets.numel() - foreground
        positive_weight = (background / (foreground + 1e-6)).clamp(1.0, 20.0)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=positive_weight
        )
        probability = torch.sigmoid(logits)
        intersection = (probability * targets).sum()
        dice = (2 * intersection + 1e-5) / (
            probability.sum() + targets.sum() + 1e-5
        )
        return 0.7 * (1 - dice) + 0.3 * bce


def dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    intersection = (prediction.bool() & target.bool()).sum().float()
    denominator = prediction.sum().float() + target.sum().float()
    if denominator.item() == 0:
        return 1.0
    return float(((2 * intersection + 1e-6) / (denominator + 1e-6)).item())


def metrics(logits: torch.Tensor, batch: NegativeBatch) -> dict[str, float]:
    prediction = logits > 0
    error_count = batch.negative_errors.sum().float()
    removal = (
        ((~prediction) & batch.negative_errors).sum().float() / error_count
        if error_count.item() > 0
        else torch.tensor(1.0, device=logits.device)
    )
    preserved = batch.nnunet_masks & ~batch.negative_errors
    preserved_count = preserved.sum().float()
    retention = (
        (prediction & preserved).sum().float() / preserved_count
        if preserved_count.item() > 0
        else torch.tensor(1.0, device=logits.device)
    )
    return {
        "correction_dice": dice(prediction, batch.targets),
        "true_ctv_dice": dice(prediction, batch.true_masks),
        "overseg_region_removal": float(removal.item()),
        "preserved_region_retention": float(retention.item()),
    }


def train_epoch(model, loader, optimizer, scaler, device, dtype, clip):
    model.train(True)
    criterion = DiceBCELoss().to(device)
    totals = {"loss": 0.0, "correction_dice": 0.0, "true_ctv_dice": 0.0}
    for batch in loader:
        batch = batch.to(device, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda", dtype=dtype
        ):
            logits = bidirectional_logits(model, batch)
            loss = criterion(logits, batch.targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip
        )
        scaler.step(optimizer)
        scaler.update()
        result = metrics(logits.detach(), batch)
        totals["loss"] += loss.item()
        totals["correction_dice"] += result["correction_dice"]
        totals["true_ctv_dice"] += result["true_ctv_dice"]
    return {key: value / len(loader) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.train(False)
    criterion = DiceBCELoss().to(device)
    keys = (
        "loss",
        "correction_dice",
        "true_ctv_dice",
        "overseg_region_removal",
        "preserved_region_retention",
    )
    totals = {key: 0.0 for key in keys}
    for batch in loader:
        batch = batch.to(device, True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda", dtype=dtype
        ):
            logits = bidirectional_logits(model, batch)
            loss = criterion(logits, batch.targets)
        result = metrics(logits, batch)
        totals["loss"] += loss.item()
        for key, value in result.items():
            totals[key] += value
    return {key: value / len(loader) for key, value in totals.items()}


def append_tsv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with path.open("a", encoding="utf-8") as file:
        if header:
            file.write("\t".join(row) + "\n")
        file.write("\t".join(str(value) for value in row.values()) + "\n")


def save_checkpoint(
    path, epoch, model, optimizer, scheduler, scaler, best, patience
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best,
            "patience_counter": patience,
        },
        path,
    )


def load_resume(path, model, optimizer, scheduler, scaler, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return state["epoch"], state["best_metric"], state["patience_counter"]


def write_prediction(
    logits: torch.Tensor, batch: NegativeBatch, output_dir: Path
) -> Path:
    probability = F.interpolate(
        torch.sigmoid(logits.float()),
        size=batch.original_hw,
        mode="bilinear",
        align_corners=False,
    )
    array = (probability[:, 0] >= 0.5).cpu().numpy().astype(np.uint8)
    reference = sitk.ReadImage(str(batch.patient_dir / "image.nii.gz"))
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"CTV_{patient_number(batch.patient_dir):03d}.nii.gz"
    sitk.WriteImage(image, str(path), useCompression=True)
    return path


@torch.no_grad()
def test_best(model, loader, device, dtype, output_dir, metrics_path):
    model.train(False)
    if metrics_path.exists():
        metrics_path.unlink()
    totals, count = {}, 0
    for batch in loader:
        name = batch.patient_dir.name
        batch = batch.to(device, True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda", dtype=dtype
        ):
            logits = bidirectional_logits(model, batch)
        result = metrics(logits, batch)
        output = write_prediction(logits, batch, output_dir)
        append_tsv(
            metrics_path,
            {"patient": name, "output": output.name, **result},
        )
        for key, value in result.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        print(
            f"[TEST] {name}->{output.name}, "
            f"correction_dice={result['correction_dice']:.4f}, "
            f"true_ctv_dice={result['true_ctv_dice']:.4f}"
        )
    means = {key: value / count for key, value in totals.items()}
    append_tsv(metrics_path, {"patient": "MEAN", "output": "", **means})
    return means


def make_loader(dataset, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_one,
        generator=generator,
        persistent_workers=workers > 0,
    )


def parse_args():
    parser = argparse.ArgumentParser("Negative-only SAM2-LoRA correction")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--test-results-dir", type=Path, default=TEST_RESULTS)
    parser.add_argument("--neg-prompt-name", default=NEG_PROMPT_NAME)
    parser.add_argument("--init-checkpoint", type=Path, default=INIT_CHECKPOINT)
    parser.add_argument("--model-config", default=MODEL_CONFIG)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--module-lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.test_results_dir.mkdir(parents=True, exist_ok=True)
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(args.init_checkpoint)
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    dtype = (
        torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    )

    all_train = list_patients(
        args.data_root / "train", args.neg_prompt_name
    )
    test_patients = list_patients(
        args.data_root / "test", args.neg_prompt_name
    )
    train_patients, validation_patients = make_split(
        all_train,
        args.validation_fraction,
        args.seed,
        args.output_root / "split.json",
    )
    print(
        f"[SPLIT] train={len(train_patients)}, "
        f"validation={len(validation_patients)}, test={len(test_patients)}"
    )
    dataset_args = dict(
        prompt_name=args.neg_prompt_name,
        input_size=args.input_size,
        window_center=args.window_center,
        window_width=args.window_width,
    )
    train_loader = make_loader(
        NegativeCorrectionDataset(train_patients, **dataset_args),
        True,
        args.num_workers,
        args.seed,
    )
    validation_loader = make_loader(
        NegativeCorrectionDataset(validation_patients, **dataset_args),
        False,
        args.num_workers,
        args.seed,
    )
    test_loader = make_loader(
        NegativeCorrectionDataset(test_patients, **dataset_args),
        False,
        args.num_workers,
        args.seed,
    )
    model, stats = build_model(args, device)
    (args.output_root / "trainable_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"[MODEL] {stats}")
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    checkpoint_dir = args.output_root / "checkpoints"
    latest = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    start, best, patience = 0, -math.inf, 0
    if args.resume and latest.is_file():
        start, best, patience = load_resume(
            latest, model, optimizer, scheduler, scaler, device
        )
        print(f"[RESUME] epoch={start}, best={best:.4f}")

    for epoch in range(start, args.max_epochs):
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            dtype,
            args.grad_clip_norm,
        )
        validation_result = evaluate(
            model, validation_loader, device, dtype
        )
        scheduler.step()
        score = validation_result["correction_dice"]
        if score > best:
            best, patience = score, 0
            save_checkpoint(
                best_path,
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                best,
                patience,
            )
        else:
            patience += 1
        save_checkpoint(
            latest,
            epoch + 1,
            model,
            optimizer,
            scheduler,
            scaler,
            best,
            patience,
        )
        append_tsv(
            args.output_root / "training_metrics.tsv",
            {
                "epoch": epoch + 1,
                "train_loss": train_result["loss"],
                "train_correction_dice": train_result["correction_dice"],
                "train_true_ctv_dice": train_result["true_ctv_dice"],
                "val_loss": validation_result["loss"],
                "val_correction_dice": score,
                "val_true_ctv_dice": validation_result["true_ctv_dice"],
                "val_overseg_region_removal": validation_result[
                    "overseg_region_removal"
                ],
                "val_preserved_region_retention": validation_result[
                    "preserved_region_retention"
                ],
                "best_val_correction_dice": best,
            },
        )
        print(
            f"[EPOCH {epoch + 1:03d}] loss={train_result['loss']:.4f}, "
            f"val_correction_dice={score:.4f}, "
            f"val_true_ctv_dice={validation_result['true_ctv_dice']:.4f}, "
            f"best={best:.4f}"
        )
        if patience >= args.patience:
            print(f"[EARLY STOP] patience={args.patience}")
            break

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"])
    print(
        f"[TEST] best epoch={state['epoch']}, "
        f"val correction Dice={state['best_metric']:.4f}"
    )
    means = test_best(
        model,
        test_loader,
        device,
        dtype,
        args.test_results_dir,
        args.output_root / "test_metrics.tsv",
    )
    print(
        f"[DONE] test correction Dice={means['correction_dice']:.4f}, "
        f"true CTV Dice={means['true_ctv_dice']:.4f}; "
        f"predictions={args.test_results_dir}"
    )


if __name__ == "__main__":
    main()
