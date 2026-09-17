#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InterfaceSAM2.py

SAM2 CTV inference backend with multi-slice mask prompts.

CLI
---
InterfaceSAM2.exe -nii CT.nii.gz -mask CTV_prompt.nii.gz -out CTV_pred.nii.gz

Prompt rules
------------
- prompt mask is a 3D NIfTI with the same geometry as CT
- every non-empty axial slice is treated as one mask prompt
- at most one prompt mask per slice (naturally enforced by the 3D mask volume)
- 1 prompt slice:
    only process that prompted slice; DO NOT propagate to other slices
- >=2 prompt slices:
    first and last prompted slices define the propagation range
    intermediate prompted slices are additional conditioning frames
    forward/backward propagation is fused by probability averaging
- slices outside the first/last prompt range are zero

Fixed files next to the script/executable
-----------------------------------------
checkpoint/
├── sam2.1_hiera_small.pt
└── Mix6_fold4_best.pth

Model config is fixed to:
configs/sam2.1/sam2.1_hiera_s.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
import pathlib
from typing import Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


def get_program_dir() -> Path:
    """Same deployment rule as the SAM1 backend."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_program_dir()
CHECKPOINT_DIR = BASE_DIR / "checkpoint"

INIT_CHECKPOINT = CHECKPOINT_DIR / "sam2.1_hiera_small.pt"
FINETUNED_CHECKPOINT = CHECKPOINT_DIR / "Mix6_fold4_best.pth"

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

INPUT_SIZE = 512
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 400.0
THRESHOLD = 0.5
DEVICE = "cuda"
AMP_DTYPE = "bfloat16"

LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


# Make local SAM2 project/package visible when running from source or packaged dir.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiment_core import configure_prompt_number_trainables
from training.model.sam2 import SAM2Train
from training.utils.data_utils import Frame, Object, VideoDatapoint, collate_fn
from bidirectional_tracking import bidirectional_outputs, single_direction_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM2 CTV segmentation with multi-slice mask prompts"
    )
    parser.add_argument(
        "-nii", "--niiPath",
        type=str,
        required=True,
        help="待分割原始CT NIfTI文件的完整路径",
    )
    parser.add_argument(
        "-mask", "--maskPath",
        type=str,
        required=True,
        help="3D mask提示NIfTI文件的完整路径",
    )
    parser.add_argument(
        "-out", "--outputPath",
        type=str,
        required=True,
        help="输出CTV预测mask NIfTI文件的完整保存路径",
    )
    return parser.parse_args()


def set_seed(seed: int = 20260616):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_windows_compatible(path: Path, map_location="cpu"):
    """
    Load checkpoints created on Linux that may contain pathlib.PosixPath objects.

    Training checkpoints can contain an `args` dictionary with PosixPath values.
    Python on Windows cannot instantiate PosixPath during unpickling, even though
    the model state_dict itself is portable. Temporarily map PosixPath to
    WindowsPath while loading this trusted local checkpoint.
    """
    if os.name != "nt":
        return torch.load(str(path), map_location=map_location)

    original_posix_path = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(str(path), map_location=map_location)
    finally:
        pathlib.PosixPath = original_posix_path


def read_nifti(path: Path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return image, array


def same_geometry(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    if a.GetSize() != b.GetSize():
        return False
    for x, y in zip(a.GetSpacing(), b.GetSpacing()):
        if abs(float(x) - float(y)) > tol:
            return False
    for x, y in zip(a.GetOrigin(), b.GetOrigin()):
        if abs(float(x) - float(y)) > tol:
            return False
    for x, y in zip(a.GetDirection(), b.GetDirection()):
        if abs(float(x) - float(y)) > tol:
            return False
    return True


def validate_inputs(
    ct_image: sitk.Image,
    ct_zyx: np.ndarray,
    prompt_image: sitk.Image,
    prompt_zyx: np.ndarray,
):
    if ct_zyx.ndim != 3:
        raise ValueError(f"输入CT必须是3D，当前shape={ct_zyx.shape}")

    if prompt_zyx.ndim != 3:
        raise ValueError(f"输入mask提示必须是3D，当前shape={prompt_zyx.shape}")

    if ct_zyx.shape != prompt_zyx.shape:
        raise ValueError(
            f"CT和mask提示shape不一致：CT={ct_zyx.shape}, mask={prompt_zyx.shape}"
        )

    if not same_geometry(ct_image, prompt_image):
        raise ValueError(
            "CT和mask提示的几何信息不一致（size/spacing/origin/direction）。"
        )


def get_prompt_slices(prompt_zyx: np.ndarray) -> list[int]:
    """
    Every non-empty axial slice is one prompt layer.
    The 3D binary prompt NIfTI naturally guarantees one mask per slice.
    """
    binary = prompt_zyx > 0
    prompt_slices = np.flatnonzero(
        binary.reshape(binary.shape[0], -1).any(axis=1)
    ).astype(int).tolist()

    if len(prompt_slices) < 1:
        raise ValueError("至少需要1层非空mask提示。")

    return prompt_slices


def window_to_uint8(
    image_2d: np.ndarray,
    wc: float = WINDOW_CENTER,
    ww: float = WINDOW_WIDTH,
) -> np.ndarray:
    image = image_2d.astype(np.float32)
    low = float(wc) - float(ww) / 2.0
    high = float(wc) + float(ww) / 2.0
    image = np.clip(image, low, high)
    return ((image - low) / (high - low + 1e-6) * 255.0).astype(np.uint8)


def build_video_datapoint(
    ct_zyx: np.ndarray,
    prompt_zyx: np.ndarray,
    input_size: int = INPUT_SIZE,
) -> VideoDatapoint:
    frames = []
    original_h = int(ct_zyx.shape[1])
    original_w = int(ct_zyx.shape[2])

    for t in range(ct_zyx.shape[0]):
        u8 = window_to_uint8(ct_zyx[t])
        rgb = np.stack([u8, u8, u8], axis=0)

        image_tensor = torch.from_numpy(rgb).float() / 255.0
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(int(input_size), int(input_size)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mask_np = (prompt_zyx[t] > 0).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(int(input_size), int(input_size)),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(torch.bool)

        frames.append(
            Frame(
                data=image_tensor,
                objects=[
                    Object(
                        object_id=1,
                        frame_index=t,
                        segment=mask_tensor,
                    )
                ],
            )
        )

    return VideoDatapoint(
        frames=frames,
        video_id=1,
        size=(original_h, original_w),
    )


def load_model_cfg_dict(model_cfg: str = MODEL_CFG):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=model_cfg)

    return OmegaConf.to_container(cfg.model, resolve=True)


def load_model(device: torch.device):
    print("\n==========================================")
    print("开始加载SAM2模型")
    print("==========================================")
    print(f"SAM2基础权重：{INIT_CHECKPOINT}")
    print(f"微调权重：{FINETUNED_CHECKPOINT}")
    print(f"模型配置：{MODEL_CFG}")
    print(f"设备：{device}")

    model_cfg_dict = load_model_cfg_dict(MODEL_CFG)
    model_cfg_dict["image_size"] = int(INPUT_SIZE)
    model_cfg_dict["freeze_image_encoder"] = False

    image_encoder_cfg = model_cfg_dict.pop("image_encoder")
    memory_attention_cfg = model_cfg_dict.pop("memory_attention")
    memory_encoder_cfg = model_cfg_dict.pop("memory_encoder")
    model_cfg_dict.pop("_target_", None)

    model = SAM2Train(
        image_encoder=instantiate(image_encoder_cfg, _recursive_=True),
        memory_attention=instantiate(memory_attention_cfg, _recursive_=True),
        memory_encoder=instantiate(memory_encoder_cfg, _recursive_=True),
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
        **model_cfg_dict,
    )

    print("\n[1/2] 加载SAM2官方基础权重...")
    init_state = torch_load_windows_compatible(INIT_CHECKPOINT, map_location="cpu")
    init_state_dict = (
        init_state["model"]
        if isinstance(init_state, dict) and "model" in init_state
        else init_state
    )
    model.load_state_dict(init_state_dict, strict=False)

    print("[2/2] 恢复LoRA结构并加载CTV微调权重...")
    stats = configure_prompt_number_trainables(
        model,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )

    fine_state = torch_load_windows_compatible(FINETUNED_CHECKPOINT, map_location="cpu")
    fine_state_dict = (
        fine_state["model"]
        if isinstance(fine_state, dict) and "model" in fine_state
        else fine_state
    )
    model.load_state_dict(fine_state_dict, strict=False)

    model.to(device)
    model.eval()

    print(f"模型加载完成：{stats}")
    return model


def build_batch(
    ct_crop_zyx: np.ndarray,
    prompt_crop_zyx: np.ndarray,
    device: torch.device,
):
    video = build_video_datapoint(
        ct_zyx=ct_crop_zyx,
        prompt_zyx=prompt_crop_zyx,
        input_size=INPUT_SIZE,
    )

    batch = collate_fn(
        [video],
        dict_key="interface_sam2",
    )
    return batch.to(device, non_blocking=True)


@torch.no_grad()
def predict_range(
    model,
    ct_crop_zyx: np.ndarray,
    prompt_crop_zyx: np.ndarray,
    prompt_frames_local: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """
    Multi-prompt case (>=2 prompt layers):
    use the same bidirectional path as validation/testing.
    """
    batch = build_batch(ct_crop_zyx, prompt_crop_zyx, device)
    core_model = model.module if hasattr(model, "module") else model

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16

    with torch.cuda.amp.autocast(
        enabled=(device.type == "cuda"),
        dtype=amp_dtype,
    ):
        outputs = bidirectional_outputs(
            core_model=core_model,
            batch=batch,
            prompt_frames=list(prompt_frames_local),
            forward_backbone_per_frame=False,
        )

    logits = torch.stack(
        [out["pred_masks_high_res"][:, 0] for out in outputs],
        dim=0,
    )[:, 0]

    logits_original = F.interpolate(
        logits.unsqueeze(1),
        size=ct_crop_zyx.shape[1:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    probability = torch.sigmoid(logits_original)
    return (
        probability >= float(THRESHOLD)
    ).cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_single_prompt_slice(
    model,
    ct_slice: np.ndarray,
    prompt_mask_slice: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    One-mask case:
    DO NOT propagate to any neighboring slice.

    A one-frame video is built and the prompt mask is used only on that frame.
    This still lets the fine-tuned SAM2 model produce the prompted-frame output,
    but there is no temporal/SI propagation because the video contains one frame.
    """
    ct_crop = ct_slice[None, ...]
    prompt_crop = (prompt_mask_slice > 0).astype(np.uint8)[None, ...]

    batch = build_batch(ct_crop, prompt_crop, device)
    core_model = model.module if hasattr(model, "module") else model

    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16

    with torch.cuda.amp.autocast(
        enabled=(device.type == "cuda"),
        dtype=amp_dtype,
    ):
        # Single frame, single direction is sufficient. There are no other frames
        # to propagate into.
        outputs = single_direction_outputs(
            core_model=core_model,
            batch=batch,
            prompt_frames=[0],
            reverse=False,
            forward_backbone_per_frame=False,
        )

    logits = outputs[0]["pred_masks_high_res"][:, 0]

    logits_original = F.interpolate(
        logits.unsqueeze(1),
        size=ct_slice.shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    probability = torch.sigmoid(logits_original)
    pred = (
        probability >= float(THRESHOLD)
    ).cpu().numpy().astype(np.uint8)

    return pred[0]


def save_prediction_like(
    pred_zyx: np.ndarray,
    reference_image: sitk.Image,
    output_path: Path,
):
    output = sitk.GetImageFromArray(pred_zyx.astype(np.uint8))
    output.CopyInformation(reference_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(output_path), useCompression=True)


def main():
    args = parse_args()

    image_path = Path(args.niiPath).expanduser()
    prompt_path = Path(args.maskPath).expanduser()
    output_path = Path(args.outputPath).expanduser()

    print("\n==========================================")
    print("InterfaceSAM2 输入输出")
    print("==========================================")
    print(f"输入CT：{image_path}")
    print(f"输入mask提示：{prompt_path}")
    print(f"输出路径：{output_path}")

    required_files = {
        "NIfTI": image_path,
        "mask prompt": prompt_path,
        "SAM2 checkpoint": INIT_CHECKPOINT,
        "finetuned checkpoint": FINETUNED_CHECKPOINT,
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}不存在：{path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ct_image, ct_array = read_nifti(image_path)
    prompt_image, prompt_array = read_nifti(prompt_path)

    validate_inputs(
        ct_image=ct_image,
        ct_zyx=ct_array,
        prompt_image=prompt_image,
        prompt_zyx=prompt_array,
    )

    prompt_slices = get_prompt_slices(prompt_array)

    print(f"\n读取到 {len(prompt_slices)} 个mask提示层：")
    for i, z in enumerate(prompt_slices, start=1):
        print(f"  Mask {i}: slice {z}")

    device_name = DEVICE
    if device_name != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA不可用，自动切换到CPU。")
        device_name = "cpu"

    device = torch.device(device_name)
    set_seed()

    model = load_model(device)

    prediction_volume = np.zeros_like(prompt_array, dtype=np.uint8)

    # ------------------------------------------------------------------
    # 1 prompt: only this slice, no propagation
    # ------------------------------------------------------------------
    if len(prompt_slices) == 1:
        z = prompt_slices[0]

        print("\n==========================================")
        print(f"仅1个mask提示：只处理slice {z}，不做传播")
        print("==========================================")

        prediction_volume[z] = predict_single_prompt_slice(
            model=model,
            ct_slice=ct_array[z],
            prompt_mask_slice=prompt_array[z],
            device=device,
        )

    # ------------------------------------------------------------------
    # >=2 prompts: first/last are mandatory bounds, middle prompts allowed
    # ------------------------------------------------------------------
    else:
        z_first = prompt_slices[0]
        z_last = prompt_slices[-1]

        ct_crop = ct_array[z_first:z_last + 1]
        prompt_crop = (
            prompt_array[z_first:z_last + 1] > 0
        ).astype(np.uint8)

        prompt_frames_local = [
            z - z_first
            for z in prompt_slices
        ]

        print("\n==========================================")
        print(
            f"开始双向传播：slice {z_first}~{z_last}，"
            f"共 {z_last-z_first+1} 层，"
            f"{len(prompt_slices)} 个mask提示层"
        )
        print("==========================================")

        pred_crop = predict_range(
            model=model,
            ct_crop_zyx=ct_crop,
            prompt_crop_zyx=prompt_crop,
            prompt_frames_local=prompt_frames_local,
            device=device,
        )

        prediction_volume[z_first:z_last + 1] = pred_crop

    print("\n正在保存结果...")
    save_prediction_like(
        pred_zyx=prediction_volume,
        reference_image=ct_image,
        output_path=output_path,
    )

    print("\n==========================================")
    print("预测完成")
    print("==========================================")
    print(f"输出文件：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(
            f"ERROR: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
