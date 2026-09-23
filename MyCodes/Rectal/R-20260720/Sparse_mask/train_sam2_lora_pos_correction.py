#!/usr/bin/env python3
"""Train SAM2-LoRA to correct nnU-Net false negatives with positive prompts.

Stage-1 objective
-----------------
Image encoder:
    - stream 1: windowed CT repeated to three channels
    - stream 2: binary nnU-Net prediction repeated to three channels
    - the two streams share one SAM2 image encoder and all LoRA weights
Prompt encoder:
    - postprocessed positive prompt only
Training target:
    - nnU-Net prediction OR raw positive error mask
Tracking:
    - forward and backward probabilities are averaged before loss/evaluation

The train patients are split once, by patient, into 90% training and 10%
validation.  The best validation checkpoint is automatically evaluated on the
independent test directory.  Test predictions are saved as CTV_XXX.nii.gz.
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


def find_project_root(start: Path = CURRENT_DIR) -> Path:
    candidates = [start] + list(start.parents)
    env_root = os.environ.get("SAM2_PROJECT_ROOT", "").strip()
    if env_root:
        candidates.insert(0, Path(env_root).resolve())
    for root in candidates:
        if (root / "sam2").is_dir() and (root / "training").is_dir():
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return root
    raise RuntimeError(
        "Cannot locate the SAM2 project root. Set SAM2_PROJECT_ROOT if needed."
    )


PROJECT_ROOT = find_project_root()

from sam2.modeling.lora import (  # noqa: E402
    LoRAConfig,
    apply_lora,
    apply_qv_lora_to_fused_qkv,
)
from training.model.sam2 import SAM2Train  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/Prompt_mask"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/"
    "PositiveCorrection/TrainResults"
)
DEFAULT_TEST_RESULTS = Path(
    "/home/wusi/SAM2/MyTrain/SAM2data/Rectal/20260720_CTV/"
    "PositiveCorrection/TestResults"
)
DEFAULT_CHECKPOINT = Path(
    "/home/wusi/SAM2/checkpoints/sam2.1_hiera_small.pt"
)
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_POS_PROMPT_NAME = "pos_erode2_top3_min50mm2_dilate2.nii.gz"
SAM2_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM2_IMAGE_STD = (0.229, 0.224, 0.225)
ERROR_METRIC_KEY = "missed_region_recall"
RETENTION_METRIC_KEY = "original_region_retention"

REQUIRED_NAMES = (
    "image.nii.gz",
    "CTV.nii.gz",
    "nnunet.nii.gz",
    "pos_raw.nii.gz",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_number(patient_dir: Path) -> int:
    match = re.search(r"(\d+)$", patient_dir.name)
    if match is None:
        raise ValueError(f"Cannot parse patient number from: {patient_dir}")
    return int(match.group(1))


def patient_sort_key(patient_dir: Path) -> tuple[int, str]:
    return patient_number(patient_dir), patient_dir.name


def list_patient_dirs(root: Path, pos_prompt_name: str) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Patient directory does not exist: {root}")
    patients = sorted(
        (path for path in root.glob("p_*") if path.is_dir()),
        key=patient_sort_key,
    )
    if not patients:
        raise RuntimeError(f"No p_* patient folders found in: {root}")

    missing: list[str] = []
    required = REQUIRED_NAMES + (pos_prompt_name,)
    for patient in patients:
        for name in required:
            if not (patient / name).is_file():
                missing.append(str(patient / name))
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing required case files:\n{preview}{suffix}")
    return patients


def make_or_load_single_split(
    patients: Sequence[Path],
    validation_fraction: float,
    seed: int,
    split_path: Path,
) -> tuple[list[Path], list[Path]]:
    patient_map = {patient.name: patient for patient in patients}
    if split_path.is_file():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        train = [patient_map[name] for name in saved["train"]]
        validation = [patient_map[name] for name in saved["validation"]]
        return train, validation

    if len(patients) < 2:
        raise RuntimeError("At least two train patients are required for a 9:1 split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")

    shuffled = list(patients)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = sorted(shuffled[:validation_count], key=patient_sort_key)
    train = sorted(shuffled[validation_count:], key=patient_sort_key)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "train": [patient.name for patient in train],
                "validation": [patient.name for patient in validation],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return train, validation


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def window_to_float(image: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError(f"Window width must be positive, got {width}")
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(image.astype(np.float32), low, high)
    return (clipped - low) / (high - low)


@dataclass
class CorrectionBatch:
    """A batch containing one variable-length 3D patient volume."""

    images: torch.Tensor
    targets: torch.Tensor
    true_masks: torch.Tensor
    nnunet_masks: torch.Tensor
    missed_masks: torch.Tensor
    positive_prompts: torch.Tensor
    case_dir: Path
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
            self.num_frames,
            dtype=torch.long,
            device=self.images.device,
        ).unsqueeze(1)

    @property
    def masks(self) -> torch.Tensor:
        return self.targets

    def to(self, device: torch.device, non_blocking: bool = False) -> "CorrectionBatch":
        return CorrectionBatch(
            images=self.images.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
            true_masks=self.true_masks.to(device, non_blocking=non_blocking),
            nnunet_masks=self.nnunet_masks.to(
                device, non_blocking=non_blocking
            ),
            missed_masks=self.missed_masks.to(
                device, non_blocking=non_blocking
            ),
            positive_prompts=self.positive_prompts.to(
                device, non_blocking=non_blocking
            ),
            case_dir=self.case_dir,
            original_hw=self.original_hw,
        )

    def pin_memory(self) -> "CorrectionBatch":
        return CorrectionBatch(
            images=self.images.pin_memory(),
            targets=self.targets.pin_memory(),
            true_masks=self.true_masks.pin_memory(),
            nnunet_masks=self.nnunet_masks.pin_memory(),
            missed_masks=self.missed_masks.pin_memory(),
            positive_prompts=self.positive_prompts.pin_memory(),
            case_dir=self.case_dir,
            original_hw=self.original_hw,
        )


class PositiveCorrectionDataset(Dataset):
    def __init__(
        self,
        patient_dirs: Sequence[Path],
        pos_prompt_name: str,
        input_size: int,
        window_center: float,
        window_width: float,
    ) -> None:
        self.patient_dirs = [Path(path) for path in patient_dirs]
        self.pos_prompt_name = pos_prompt_name
        self.input_size = int(input_size)
        self.window_center = float(window_center)
        self.window_width = float(window_width)

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, index: int) -> CorrectionBatch:
        patient_dir = self.patient_dirs[index]
        paths = {
            "image": patient_dir / "image.nii.gz",
            "ctv": patient_dir / "CTV.nii.gz",
            "nnunet": patient_dir / "nnunet.nii.gz",
            "pos_raw": patient_dir / "pos_raw.nii.gz",
            "pos_prompt": patient_dir / self.pos_prompt_name,
        }
        itk_images = {name: sitk.ReadImage(str(path)) for name, path in paths.items()}
        reference = itk_images["image"]
        for name, itk_image in itk_images.items():
            if not same_geometry(reference, itk_image):
                raise ValueError(
                    f"Geometry mismatch in {patient_dir}: image vs {name}"
                )

        arrays = {
            name: sitk.GetArrayFromImage(itk_image)
            for name, itk_image in itk_images.items()
        }
        shapes = {name: array.shape for name, array in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Array shape mismatch in {patient_dir}: {shapes}")

        ct = window_to_float(
            arrays["image"], self.window_center, self.window_width
        )
        nnunet = arrays["nnunet"] > 0
        pos_raw = arrays["pos_raw"] > 0
        pos_prompt = arrays["pos_prompt"] > 0
        true_mask = arrays["ctv"] > 0
        correction_target = np.logical_or(nnunet, pos_raw)

        ct_tensor = torch.from_numpy(ct).unsqueeze(1)
        nnunet_tensor = torch.from_numpy(nnunet.astype(np.float32)).unsqueeze(1)
        target_tensor = torch.from_numpy(
            correction_target.astype(np.float32)
        ).unsqueeze(1)
        true_tensor = torch.from_numpy(true_mask.astype(np.float32)).unsqueeze(1)
        pos_raw_tensor = torch.from_numpy(pos_raw.astype(np.float32)).unsqueeze(1)
        prompt_tensor = torch.from_numpy(pos_prompt.astype(np.float32)).unsqueeze(1)

        resize_to = (self.input_size, self.input_size)
        ct_tensor = F.interpolate(
            ct_tensor,
            size=resize_to,
            mode="bilinear",
            align_corners=False,
        )
        nnunet_tensor = F.interpolate(
            nnunet_tensor, size=resize_to, mode="nearest"
        )
        target_tensor = F.interpolate(
            target_tensor, size=resize_to, mode="nearest"
        )
        true_tensor = F.interpolate(true_tensor, size=resize_to, mode="nearest")
        pos_raw_tensor = F.interpolate(
            pos_raw_tensor, size=resize_to, mode="nearest"
        )
        prompt_tensor = F.interpolate(
            prompt_tensor, size=resize_to, mode="nearest"
        )

        ct_rgb = ct_tensor.repeat(1, 3, 1, 1)
        nnunet_rgb = nnunet_tensor.repeat(1, 3, 1, 1)
        mean = torch.tensor(SAM2_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(SAM2_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
        ct_rgb = (ct_rgb - mean) / std
        nnunet_rgb = (nnunet_rgb - mean) / std
        dual_stream_input = torch.cat((ct_rgb, nnunet_rgb), dim=1)
        original_hw = (int(ct.shape[1]), int(ct.shape[2]))

        return CorrectionBatch(
            images=dual_stream_input.float(),
            targets=target_tensor.bool(),
            true_masks=true_tensor.bool(),
            nnunet_masks=nnunet_tensor.bool(),
            missed_masks=pos_raw_tensor.bool(),
            positive_prompts=prompt_tensor.float(),
            case_dir=patient_dir,
            original_hw=original_hw,
        )


def collate_one(items: list[CorrectionBatch]) -> CorrectionBatch:
    if len(items) != 1:
        raise ValueError("Variable-length volume training requires batch_size=1")
    return items[0]


class DualStreamSAM2Train(SAM2Train):
    """SAM2Train with a shared CT/nnU-Net image encoder and gated FPN fusion."""

    def __init__(self, *args, fusion_levels: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initial_gate = math.log(0.1 / 0.9)
        self.dual_stream_feature_gates = nn.Parameter(
            torch.full((int(fusion_levels),), initial_gate, dtype=torch.float32)
        )

    def forward_image(self, img_batch: torch.Tensor) -> dict:
        if img_batch.ndim != 4 or img_batch.shape[1] != 6:
            raise ValueError(
                "DualStreamSAM2Train expects [N, 6, H, W] input "
                f"(CT RGB + nnU-Net RGB), got {tuple(img_batch.shape)}"
            )
        ct_output = super().forward_image(img_batch[:, :3])
        nnunet_output = super().forward_image(img_batch[:, 3:])

        ct_fpn = ct_output["backbone_fpn"]
        nnunet_fpn = nnunet_output["backbone_fpn"]
        if len(ct_fpn) != len(nnunet_fpn):
            raise RuntimeError("CT and nnU-Net FPN outputs have different lengths")
        if len(ct_fpn) != self.dual_stream_feature_gates.numel():
            raise RuntimeError(
                "Unexpected FPN level count: "
                f"{len(ct_fpn)} vs configured "
                f"{self.dual_stream_feature_gates.numel()}"
            )

        fused = dict(ct_output)
        gates = torch.sigmoid(self.dual_stream_feature_gates)
        fused["backbone_fpn"] = [
            ct_feature + gates[level].to(ct_feature.dtype) * nnunet_feature
            for level, (ct_feature, nnunet_feature) in enumerate(
                zip(ct_fpn, nnunet_fpn)
            )
        ]
        if (
            "vision_features" in ct_output
            and "vision_features" in nnunet_output
        ):
            fused["vision_features"] = (
                ct_output["vision_features"]
                + gates[-1].to(ct_output["vision_features"].dtype)
                * nnunet_output["vision_features"]
            )
        # Positional encodings depend on spatial coordinates, not input modality.
        fused["vision_pos_enc"] = ct_output["vision_pos_enc"]
        return fused


def load_model_config(model_cfg: str) -> dict:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module("sam2", version_base="1.2"):
        config = compose(config_name=model_cfg)
    return OmegaConf.to_container(config.model, resolve=True)


def configure_trainable_parameters(
    model: DualStreamSAM2Train,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    common = {
        "r": int(lora_r),
        "alpha": int(lora_alpha),
        "dropout": float(lora_dropout),
        "freeze_base_model": False,
    }
    image_lora = apply_qv_lora_to_fused_qkv(
        model,
        LoRAConfig(
            target_modules=("qkv",),
            target_prefixes=("image_encoder",),
            **common,
        ),
    )
    memory_attention_lora = apply_lora(
        model,
        LoRAConfig(
            target_modules=("q_proj", "v_proj"),
            target_prefixes=("memory_attention",),
            **common,
        ),
    )
    memory_encoder_lora = apply_lora(
        model,
        LoRAConfig(
            target_modules=("pwconv1", "pwconv2"),
            target_prefixes=("memory_encoder.fuser",),
            **common,
        ),
    )

    for module_name in ("sam_prompt_encoder", "sam_mask_decoder"):
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.dual_stream_feature_gates.requires_grad = True

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "image_encoder_lora_layers": int(image_lora),
        "memory_attention_lora_layers": int(memory_attention_lora),
        "memory_encoder_lora_layers": int(memory_encoder_lora),
        "trainable_params": int(trainable),
        "total_params": int(total),
    }


def build_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DualStreamSAM2Train, dict[str, int]]:
    model_config = load_model_config(args.model_cfg)
    model_config["image_size"] = int(args.input_size)
    model_config["freeze_image_encoder"] = False
    # The positive mask must pass through the prompt encoder and mask decoder.
    model_config["use_mask_input_as_output_without_sam"] = False
    # Diagnostic run showed that unsupervised object-score hard gating suppresses
    # frames with zero gradient, while the IoU head selects huge multimasks.
    model_config["pred_obj_scores"] = False
    model_config["fixed_no_obj_ptr"] = False
    model_config["multimask_output_in_sam"] = False
    model_config["multimask_output_for_tracking"] = False
    image_encoder_config = model_config.pop("image_encoder")
    memory_attention_config = model_config.pop("memory_attention")
    memory_encoder_config = model_config.pop("memory_encoder")
    model_config.pop("_target_", None)

    model = DualStreamSAM2Train(
        image_encoder=instantiate(image_encoder_config, _recursive_=True),
        memory_attention=instantiate(memory_attention_config, _recursive_=True),
        memory_encoder=instantiate(memory_encoder_config, _recursive_=True),
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        prob_to_sample_from_gt_for_train=0.0,
        num_frames_to_correct_for_train=1,
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        add_all_frames_to_correct_as_cond=False,
        num_correction_pt_per_frame=0,
        rand_init_cond_frames_for_train=False,
        rand_init_cond_frames_for_eval=False,
        fusion_levels=3,
        **model_config,
    )

    checkpoint = torch.load(str(args.init_checkpoint), map_location="cpu")
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("dual_stream_feature_gates")
    ]
    if unexpected:
        print(f"[WARN] unexpected pretrained keys: {unexpected[:20]}")

    stats = configure_trainable_parameters(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if stats["image_encoder_lora_layers"] == 0:
        raise RuntimeError("No image encoder LoRA layers were installed")
    if stats["memory_attention_lora_layers"] == 0:
        raise RuntimeError("No memory attention LoRA layers were installed")
    if stats["memory_encoder_lora_layers"] == 0:
        raise RuntimeError("No memory encoder LoRA layers were installed")
    return model.to(device), stats


def build_optimizer(
    model: nn.Module,
    adapter_lr: float,
    module_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    adapter_parameters: list[nn.Parameter] = []
    module_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name or name == "dual_stream_feature_gates":
            adapter_parameters.append(parameter)
        else:
            module_parameters.append(parameter)
    if not adapter_parameters or not module_parameters:
        raise RuntimeError(
            "Expected non-empty adapter and prompt/mask-decoder parameter groups"
        )
    return torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(adapter_lr)},
            {"params": module_parameters, "lr": float(module_lr)},
        ],
        weight_decay=float(weight_decay),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def make_schedule(base_lr: float):
        min_factor = float(min_lr) / max(float(base_lr), 1e-12)

        def schedule(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(
                1, max_epochs - warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_factor, cosine)

        return schedule

    schedules = [
        make_schedule(float(group["lr"])) for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=schedules,
    )


def positive_prompt_frames(batch: CorrectionBatch) -> list[int]:
    nonempty = batch.positive_prompts.flatten(1).any(dim=1)
    indices = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
    # A zero mask on frame 0 is a safe conditioning fallback for no-error cases.
    return [int(index) for index in indices] if indices else [0]


def precompute_backbone(model: DualStreamSAM2Train, batch: CorrectionBatch) -> dict:
    return model.forward_image(batch.flat_img_batch)


def track_direction(
    model: DualStreamSAM2Train,
    base_backbone: dict,
    batch: CorrectionBatch,
    prompt_frames: Sequence[int],
    reverse: bool,
) -> list[dict]:
    _, vision_features, vision_positions, feature_sizes = (
        model._prepare_backbone_features(base_backbone)
    )
    prompt_frames = sorted(
        set(int(frame) for frame in prompt_frames),
        reverse=reverse,
    )
    prompt_set = set(prompt_frames)
    remaining = [
        frame
        for frame in range(batch.num_frames)
        if frame not in prompt_set
    ]
    remaining.sort(reverse=reverse)
    processing_order = prompt_frames + remaining

    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    all_outputs: dict[int, dict] = {}
    for frame in processing_order:
        image_ids = batch.flat_obj_to_img_idx[frame]
        current_features = [
            feature[:, image_ids] for feature in vision_features
        ]
        current_positions = [
            position[:, image_ids] for position in vision_positions
        ]
        mask_input = (
            batch.positive_prompts[frame].unsqueeze(1)
            if frame in prompt_set
            else None
        )
        current_output = model.track_step(
            frame_idx=frame,
            is_init_cond_frame=frame in prompt_set,
            current_vision_feats=current_features,
            current_vision_pos_embeds=current_positions,
            feat_sizes=feature_sizes,
            point_inputs=None,
            mask_inputs=mask_input,
            gt_masks=batch.targets[frame].unsqueeze(1),
            frames_to_add_correction_pt=[],
            output_dict=output_dict,
            num_frames=batch.num_frames,
            track_in_reverse=reverse,
            prev_sam_mask_logits=None,
        )
        if frame in prompt_set:
            output_dict["cond_frame_outputs"][frame] = current_output
        else:
            output_dict["non_cond_frame_outputs"][frame] = current_output
        all_outputs[frame] = current_output

    return [
        {
            key: value
            for key, value in all_outputs[frame].items()
            if key != "obj_ptr"
        }
        for frame in range(batch.num_frames)
    ]


def bidirectional_logits(
    model: DualStreamSAM2Train,
    batch: CorrectionBatch,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    prompt_frames = positive_prompt_frames(batch)
    backbone = precompute_backbone(model, batch)
    forward_outputs = track_direction(
        model,
        backbone,
        batch,
        prompt_frames=prompt_frames,
        reverse=False,
    )
    backward_outputs = track_direction(
        model,
        backbone,
        batch,
        prompt_frames=prompt_frames,
        reverse=True,
    )
    forward_logits = torch.stack(
        [output["pred_masks_high_res"][:, 0] for output in forward_outputs],
        dim=0,
    )
    backward_logits = torch.stack(
        [output["pred_masks_high_res"][:, 0] for output in backward_outputs],
        dim=0,
    )
    probability = 0.5 * (
        torch.sigmoid(forward_logits) + torch.sigmoid(backward_logits)
    )
    fused_logits = torch.logit(probability.clamp(1e-4, 1.0 - 1e-4))
    if not return_debug:
        return fused_logits

    def direction_stats(outputs: Sequence[dict], prefix: str) -> dict[str, float]:
        object_scores = torch.cat(
            [
                output["multistep_object_score_logits"][-1]
                .detach()
                .float()
                .flatten()
                for output in outputs
            ]
        )
        iou_scores = torch.cat(
            [
                output["multistep_pred_ious"][-1]
                .detach()
                .float()
                .flatten()
                for output in outputs
            ]
        )
        multimasks = torch.cat(
            [
                output["multistep_pred_multimasks_high_res"][-1]
                .detach()
                .float()
                for output in outputs
            ],
            dim=0,
        )
        selected = torch.stack(
            [
                output["pred_masks_high_res"][:, 0].detach().float()
                for output in outputs
            ]
        )
        return {
            f"{prefix}_object_score_min": float(object_scores.min().item()),
            f"{prefix}_object_score_mean": float(object_scores.mean().item()),
            f"{prefix}_object_score_max": float(object_scores.max().item()),
            f"{prefix}_object_present_fraction": float(
                (object_scores > 0).float().mean().item()
            ),
            f"{prefix}_iou_min": float(iou_scores.min().item()),
            f"{prefix}_iou_mean": float(iou_scores.mean().item()),
            f"{prefix}_iou_max": float(iou_scores.max().item()),
            f"{prefix}_multimask_foreground_fraction": float(
                (multimasks > 0).float().mean().item()
            ),
            f"{prefix}_selected_foreground_fraction": float(
                (selected > 0).float().mean().item()
            ),
            f"{prefix}_selected_logit_mean": float(selected.mean().item()),
            f"{prefix}_selected_logit_max": float(selected.max().item()),
        }

    debug = {
        "num_frames": float(batch.num_frames),
        "num_prompt_frames": float(len(prompt_frames)),
        "prompt_foreground_fraction": float(
            (batch.positive_prompts > 0).float().mean().item()
        ),
        "target_foreground_fraction": float(
            batch.targets.float().mean().item()
        ),
        "nnunet_foreground_fraction": float(
            batch.nnunet_masks.float().mean().item()
        ),
        "fused_foreground_fraction": float(
            (fused_logits > 0).float().mean().item()
        ),
        "fused_logit_min": float(fused_logits.detach().float().min().item()),
        "fused_logit_mean": float(fused_logits.detach().float().mean().item()),
        "fused_logit_max": float(fused_logits.detach().float().max().item()),
        **direction_stats(forward_outputs, "forward"),
        **direction_stats(backward_outputs, "backward"),
    }
    return fused_logits, debug


def gradient_debug_stats(model: nn.Module) -> dict[str, float]:
    groups = {
        "image_lora": lambda name: name.startswith("image_encoder.")
        and "lora_" in name,
        "memory_attention_lora": lambda name: name.startswith(
            "memory_attention."
        )
        and "lora_" in name,
        "memory_encoder_lora": lambda name: name.startswith("memory_encoder.")
        and "lora_" in name,
        "prompt_encoder": lambda name: name.startswith("sam_prompt_encoder."),
        "mask_decoder": lambda name: name.startswith("sam_mask_decoder."),
        "object_score_head": lambda name: "pred_obj_score_head" in name,
        "feature_gates": lambda name: name == "dual_stream_feature_gates",
    }
    stats: dict[str, float] = {}
    named_parameters = list(model.named_parameters())
    for group_name, matches in groups.items():
        gradients = [
            parameter.grad.detach().float()
            for name, parameter in named_parameters
            if parameter.requires_grad
            and matches(name)
            and parameter.grad is not None
        ]
        squared_norm = sum(
            (gradient * gradient).sum() for gradient in gradients
        )
        stats[f"{group_name}_grad_tensor_count"] = float(len(gradients))
        stats[f"{group_name}_grad_norm"] = (
            float(torch.sqrt(squared_norm).item()) if gradients else 0.0
        )
    return stats


class DiceBCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        foreground = targets.sum()
        background = targets.numel() - foreground
        positive_weight = (background / (foreground + 1e-6)).clamp(1.0, 20.0)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_weight,
        )
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum()
        denominator = probabilities.sum() + targets.sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (
            denominator + 1e-5
        )
        return 0.7 * dice_loss + 0.3 * bce


def binary_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().float()
    denominator = prediction.sum().float() + target.sum().float()
    if float(denominator.item()) == 0.0:
        return 1.0
    return float(((2.0 * intersection + 1e-6) / (denominator + 1e-6)).item())


def calculate_metrics(
    logits: torch.Tensor,
    batch: CorrectionBatch,
) -> dict[str, float]:
    prediction = logits > 0
    correction_target = batch.targets
    true_target = batch.true_masks
    nnunet_region = batch.nnunet_masks
    missed_region = batch.missed_masks

    missed_denominator = missed_region.sum().float()
    missed_recall = (
        float((prediction & missed_region).sum().float().div(missed_denominator).item())
        if float(missed_denominator.item()) > 0
        else 1.0
    )
    nnunet_denominator = nnunet_region.sum().float()
    original_retention = (
        float(
            (prediction & nnunet_region)
            .sum()
            .float()
            .div(nnunet_denominator)
            .item()
        )
        if float(nnunet_denominator.item()) > 0
        else 1.0
    )
    return {
        "correction_dice": binary_dice(prediction, correction_target),
        "true_ctv_dice": binary_dice(prediction, true_target),
        ERROR_METRIC_KEY: missed_recall,
        RETENTION_METRIC_KEY: original_retention,
    }


def train_one_epoch(
    model: DualStreamSAM2Train,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_clip_norm: float,
    epoch: int,
    debug_epochs: int,
    debug_batches: int,
    debug_path: Path,
) -> dict[str, float]:
    model.train(True)
    criterion = DiceBCELoss().to(device)
    totals = {
        "loss": 0.0,
        "correction_dice": 0.0,
        "true_ctv_dice": 0.0,
    }
    count = 0

    for batch_index, batch in enumerate(loader):
        patient_name = batch.case_dir.name
        batch = batch.to(device, non_blocking=True)
        collect_debug = epoch < debug_epochs and batch_index < debug_batches
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda",
            dtype=amp_dtype,
        ):
            inference_result = bidirectional_logits(
                model,
                batch,
                return_debug=collect_debug,
            )
            if collect_debug:
                logits, debug_stats = inference_result
            else:
                logits = inference_result
                debug_stats = {}
            loss = criterion(logits, batch.targets)

        scaler.scale(loss).backward()
        if grad_clip_norm > 0 or collect_debug:
            scaler.unscale_(optimizer)
        if collect_debug:
            debug_stats.update(gradient_debug_stats(model))
            debug_row = {
                "epoch": epoch + 1,
                "batch": batch_index,
                "patient": patient_name,
                "loss": float(loss.detach().item()),
                **debug_stats,
            }
            append_metrics(debug_path, debug_row)
            if debug_stats["fused_foreground_fraction"] < 1e-6:
                print(
                    f"[DEBUG WARNING] {patient_name}: fused prediction is empty; "
                    f"forward object-present="
                    f"{debug_stats['forward_object_present_fraction']:.3f}, "
                    f"backward object-present="
                    f"{debug_stats['backward_object_present_fraction']:.3f}, "
                    f"target foreground="
                    f"{debug_stats['target_foreground_fraction']:.6f}"
                )
            if (
                debug_stats["object_score_head_grad_norm"] == 0.0
                and min(
                    debug_stats["forward_object_present_fraction"],
                    debug_stats["backward_object_present_fraction"],
                )
                < 1.0
            ):
                print(
                    f"[DEBUG WARNING] {patient_name}: object-score gating removed "
                    "at least one frame, but object-score head gradient is zero."
                )
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=float(grad_clip_norm),
            )
        scaler.step(optimizer)
        scaler.update()

        metrics = calculate_metrics(logits.detach(), batch)
        totals["loss"] += float(loss.detach().item())
        totals["correction_dice"] += metrics["correction_dice"]
        totals["true_ctv_dice"] += metrics["true_ctv_dice"]
        count += 1

    if count == 0:
        raise RuntimeError("Training loader produced no batches")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: DualStreamSAM2Train,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.train(False)
    criterion = DiceBCELoss().to(device)
    metric_names = (
        "loss",
        "correction_dice",
        "true_ctv_dice",
        ERROR_METRIC_KEY,
        RETENTION_METRIC_KEY,
    )
    totals = {name: 0.0 for name in metric_names}
    count = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda",
            dtype=amp_dtype,
        ):
            logits = bidirectional_logits(model, batch)
            loss = criterion(logits, batch.targets)
        metrics = calculate_metrics(logits, batch)
        totals["loss"] += float(loss.item())
        for name, value in metrics.items():
            totals[name] += value
        count += 1

    if count == 0:
        raise RuntimeError("Evaluation loader produced no batches")
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_metric: float,
    patience_counter: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": float(best_metric),
            "patience_counter": int(patience_counter),
            "args": {key: str(value) for key, value in vars(args).items()},
        },
        str(path),
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(str(path), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]),
        float(checkpoint["best_metric"]),
        int(checkpoint.get("patience_counter", 0)),
    )


def append_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as file:
        if write_header:
            file.write("\t".join(row.keys()) + "\n")
        file.write("\t".join(str(value) for value in row.values()) + "\n")


def write_prediction(
    logits: torch.Tensor,
    batch: CorrectionBatch,
    output_dir: Path,
) -> Path:
    probability = torch.sigmoid(logits.float())
    probability = F.interpolate(
        probability,
        size=batch.original_hw,
        mode="bilinear",
        align_corners=False,
    )
    prediction = (probability[:, 0] >= 0.5).cpu().numpy().astype(np.uint8)

    reference = sitk.ReadImage(str(batch.case_dir / "image.nii.gz"))
    output_image = sitk.GetImageFromArray(prediction)
    output_image.CopyInformation(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"CTV_{patient_number(batch.case_dir):03d}.nii.gz"
    sitk.WriteImage(output_image, str(output_path), useCompression=True)
    return output_path


@torch.no_grad()
def test_and_save(
    model: DualStreamSAM2Train,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    output_dir: Path,
    metrics_path: Path,
) -> dict[str, float]:
    model.train(False)
    totals: dict[str, float] = {}
    count = 0
    if metrics_path.exists():
        metrics_path.unlink()

    for batch in loader:
        case_name = batch.case_dir.name
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda",
            dtype=amp_dtype,
        ):
            logits = bidirectional_logits(model, batch)
        metrics = calculate_metrics(logits, batch)
        output_path = write_prediction(logits, batch, output_dir)
        row: dict[str, object] = {
            "patient": case_name,
            "output": output_path.name,
            **metrics,
        }
        append_metrics(metrics_path, row)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
        print(
            f"[TEST] {case_name} -> {output_path.name}; "
            f"correction_dice={metrics['correction_dice']:.4f}, "
            f"true_ctv_dice={metrics['true_ctv_dice']:.4f}"
        )

    if count == 0:
        raise RuntimeError("Test loader produced no batches")
    means = {name: value / count for name, value in totals.items()}
    append_metrics(
        metrics_path,
        {
            "patient": "MEAN",
            "output": "",
            **means,
        },
    )
    return means


def make_loader(
    dataset: Dataset,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_one,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train positive-only SAM2-LoRA nnU-Net false-negative correction "
            "and automatically test the best checkpoint."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--test-results-dir",
        type=Path,
        default=DEFAULT_TEST_RESULTS,
    )
    parser.add_argument(
        "--pos-prompt-name",
        type=str,
        default=DEFAULT_POS_PROMPT_NAME,
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
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
    parser.add_argument(
        "--debug-epochs",
        type=int,
        default=3,
        help="Collect detailed diagnostics during the first N epochs.",
    )
    parser.add_argument(
        "--debug-batches",
        type=int,
        default=2,
        help="Collect diagnostics for the first N train cases per debug epoch.",
    )
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initial SAM2 checkpoint does not exist: {args.init_checkpoint}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.test_results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    )

    all_train_patients = list_patient_dirs(
        args.data_root / "train",
        args.pos_prompt_name,
    )
    test_patients = list_patient_dirs(
        args.data_root / "test",
        args.pos_prompt_name,
    )
    train_patients, validation_patients = make_or_load_single_split(
        all_train_patients,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split_path=args.output_root / "split.json",
    )
    print(
        f"[SPLIT] train={len(train_patients)}, "
        f"validation={len(validation_patients)}, test={len(test_patients)}"
    )

    dataset_kwargs = {
        "pos_prompt_name": args.pos_prompt_name,
        "input_size": args.input_size,
        "window_center": args.window_center,
        "window_width": args.window_width,
    }
    train_dataset = PositiveCorrectionDataset(
        train_patients,
        **dataset_kwargs,
    )
    validation_dataset = PositiveCorrectionDataset(
        validation_patients,
        **dataset_kwargs,
    )
    test_dataset = PositiveCorrectionDataset(
        test_patients,
        **dataset_kwargs,
    )
    train_loader = make_loader(
        train_dataset,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = make_loader(
        test_dataset,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model, trainable_stats = build_model(args, device)
    (args.output_root / "trainable_stats.json").write_text(
        json.dumps(trainable_stats, indent=2),
        encoding="utf-8",
    )
    print(f"[MODEL] {trainable_stats}")

    optimizer = build_optimizer(
        model,
        adapter_lr=args.adapter_lr,
        module_lr=args.module_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    checkpoint_dir = args.output_root / "checkpoints"
    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best.pth"
    metrics_path = args.output_root / "training_metrics.tsv"
    start_epoch = 0
    best_metric = -math.inf
    patience_counter = 0
    if args.resume and latest_path.is_file():
        start_epoch, best_metric, patience_counter = load_training_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        print(
            f"[RESUME] epoch={start_epoch}, best={best_metric:.4f}, "
            f"patience={patience_counter}"
        )

    for epoch in range(start_epoch, args.max_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_dtype,
            grad_clip_norm=args.grad_clip_norm,
            epoch=epoch,
            debug_epochs=args.debug_epochs,
            debug_batches=args.debug_batches,
            debug_path=args.output_root / "debug_diagnostics.tsv",
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            amp_dtype,
        )
        scheduler.step()

        validation_score = validation_metrics["correction_dice"]
        improved = validation_score > best_metric
        if improved:
            best_metric = validation_score
            patience_counter = 0
            save_checkpoint(
                best_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                patience_counter=patience_counter,
                args=args,
            )
        else:
            patience_counter += 1

        save_checkpoint(
            latest_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_metric,
            patience_counter=patience_counter,
            args=args,
        )
        row = {
            "epoch": epoch + 1,
            "adapter_lr": optimizer.param_groups[0]["lr"],
            "module_lr": optimizer.param_groups[1]["lr"],
            "train_loss": train_metrics["loss"],
            "train_correction_dice": train_metrics["correction_dice"],
            "train_true_ctv_dice": train_metrics["true_ctv_dice"],
            "val_loss": validation_metrics["loss"],
            "val_correction_dice": validation_metrics["correction_dice"],
            "val_true_ctv_dice": validation_metrics["true_ctv_dice"],
            f"val_{ERROR_METRIC_KEY}": validation_metrics[ERROR_METRIC_KEY],
            f"val_{RETENTION_METRIC_KEY}": validation_metrics[
                RETENTION_METRIC_KEY
            ],
            "best_val_correction_dice": best_metric,
        }
        append_metrics(metrics_path, row)
        print(
            f"[EPOCH {epoch + 1:03d}/{args.max_epochs}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_correction_dice={train_metrics['correction_dice']:.4f}, "
            f"val_correction_dice={validation_score:.4f}, "
            f"val_true_ctv_dice={validation_metrics['true_ctv_dice']:.4f}, "
            f"best={best_metric:.4f}"
        )
        if patience_counter >= args.patience:
            print(
                f"[EARLY STOP] no validation improvement for "
                f"{args.patience} epochs"
            )
            break

    if not best_path.is_file():
        raise RuntimeError(f"Best checkpoint was not created: {best_path}")
    best_checkpoint = torch.load(str(best_path), map_location=device)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    print(
        f"[TEST] loaded best checkpoint from epoch "
        f"{best_checkpoint['epoch']} with validation correction Dice "
        f"{best_checkpoint['best_metric']:.4f}"
    )
    test_means = test_and_save(
        model,
        test_loader,
        device,
        amp_dtype,
        output_dir=args.test_results_dir,
        metrics_path=args.output_root / "test_metrics.tsv",
    )
    print(
        f"[DONE] bidirectional test complete; "
        f"mean correction Dice={test_means['correction_dice']:.4f}, "
        f"mean true CTV Dice={test_means['true_ctv_dice']:.4f}; "
        f"predictions={args.test_results_dir}"
    )


if __name__ == "__main__":
    main()
