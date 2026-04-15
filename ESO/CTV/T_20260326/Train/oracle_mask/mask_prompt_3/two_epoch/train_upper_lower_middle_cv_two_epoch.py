#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2 finetuning with 3 mask prompts:
  upper + lower + middle(best from mask_prompt_3.xlsx)

Example:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_upper_lower_middle_cv_two_epoch.py
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ---- robust import path setup ----
CURRENT_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "training").is_dir() and (p / "sam2").is_dir():
            return p
    env_root = os.environ.get("SAM2_PROJECT_ROOT", "").strip()
    if env_root:
        p = Path(env_root).resolve()
        if (p / "training").is_dir() and (p / "sam2").is_dir():
            return p
    raise RuntimeError(
        "Cannot locate SAM2 project root from script path. "
        "Set SAM2_PROJECT_ROOT to a directory containing 'training' and 'sam2'."
    )


PROJECT_ROOT = _find_project_root(CURRENT_DIR)
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2_train_upper_lower_middle import SAM2TrainUpperLowerMiddleMask
from training.loss_fns import MultiStepMultiMasksAndIous
from training.utils.data_utils import Frame, Object, VideoDatapoint, collate_fn


# ================= Default Paths (edit here) =================
DEFAULT_TRAIN_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii/train_nii")
DEFAULT_PROMPT3_XLSX = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/zero-shot/oracle_mask/mask_prompt_3/prompt_layer_search3.xlsx")
DEFAULT_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/oracle_mask/mask_prompt_3/two_epoch/TrainResult")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_PRETRAINED_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_to_uint8(img2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def patient_sort_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value)).strip()
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value)).strip()
        return str(value).strip()
    text = str(value).strip()
    if text.endswith(".0"):
        numeric = text[:-2]
        if numeric.isdigit():
            return numeric
    return text


def patient_id_from_folder(pdir: Path) -> str:
    m = re.search(r"(\d+)", pdir.name)
    if m is None:
        raise ValueError(f"Cannot parse patient id from folder name: {pdir.name}")
    return f"CTV_{int(m.group(1)):03d}"


def patient_video_num_from_id(patient_id: str) -> int:
    m = re.search(r"(\d+)$", str(patient_id))
    if m is None:
        raise ValueError(f"Cannot parse numeric id from Patient_ID: {patient_id}")
    return int(m.group(1))


def find_sheet_with_columns(excel_path: Path, required_columns):
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    sheets = pd.read_excel(excel_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        if all(col in df.columns for col in required_columns):
            print(f"[INFO] Using sheet '{sheet_name}' from {excel_path.name}")
            return df.copy()
    raise ValueError(
        f"Cannot find required columns {list(required_columns)} in any sheet of {excel_path}"
    )


def load_middle_prompt_map(prompt3_xlsx: Path):
    """
    Read middle prompt IDs from mask_prompt_3.py output excel.
    Expected columns:
      - Patient_ID
      - Best_Prompt_Slice_ID
    """
    df = find_sheet_with_columns(prompt3_xlsx, ["Patient_ID", "Best_Prompt_Slice_ID"])
    df = df[["Patient_ID", "Best_Prompt_Slice_ID"]].copy()
    df["Patient_ID"] = df["Patient_ID"].apply(normalize_id)
    df["Best_Prompt_Slice_ID"] = df["Best_Prompt_Slice_ID"].apply(normalize_id)

    df = df[(df["Patient_ID"] != "") & (df["Best_Prompt_Slice_ID"] != "")].copy()

    prompt_map = {}
    for _, row in df.iterrows():
        pid = str(row["Patient_ID"])
        try:
            vid = patient_video_num_from_id(pid)
            mid = int(float(row["Best_Prompt_Slice_ID"]))
        except Exception as exc:
            raise ValueError(f"Invalid row in prompt table: {row.to_dict()}") from exc

        if vid in prompt_map and prompt_map[vid] != mid:
            raise ValueError(
                f"Duplicate Patient_ID with different middle slice: {pid} -> {prompt_map[vid]} vs {mid}"
            )
        prompt_map[vid] = mid

    if len(prompt_map) == 0:
        raise ValueError(f"No valid middle prompt records loaded from: {prompt3_xlsx}")

    print(f"[INFO] Loaded middle prompt IDs for {len(prompt_map)} patients")
    return prompt_map


class UpperLowerMiddleVolumeDataset(Dataset):
    """
    Build full-volume videos from 3D NIfTI:
    - each item is one patient volume (no clip split)
    - one object (CTV) per frame
    - middle prompt index is preloaded from prompt3 table and later consumed by model
    """

    def __init__(
        self,
        patient_dirs,
        middle_prompt_by_video_id,
        image_name="image.nii.gz",
        mask_name="CTV.nii.gz",
        window_center=40.0,
        window_width=400.0,
        input_size=1024,
        strict_middle=True,
    ):
        self.patient_dirs = list(patient_dirs)
        self.middle_prompt_by_video_id = dict(middle_prompt_by_video_id)
        self.image_name = image_name
        self.mask_name = mask_name
        self.window_center = float(window_center)
        self.window_width = float(window_width)
        self.input_size = int(input_size)
        self.strict_middle = bool(strict_middle)

        self.samples = []
        missing_middle = []

        for pdir in self.patient_dirs:
            img_path = pdir / self.image_name
            gt_path = pdir / self.mask_name
            if not img_path.exists() or not gt_path.exists():
                continue

            patient_id = patient_id_from_folder(pdir)
            video_id = patient_video_num_from_id(patient_id)

            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
            gt = (gt > 0).astype(np.uint8)
            pos = np.where(gt.sum(axis=(1, 2)) > 0)[0]
            if len(pos) == 0:
                continue

            if video_id not in self.middle_prompt_by_video_id:
                missing_middle.append(patient_id)
                if self.strict_middle:
                    continue

            self.samples.append(
                {
                    "pdir": pdir,
                    "patient_id": patient_id,
                    "video_id": video_id,
                }
            )

        if self.strict_middle and len(missing_middle) > 0:
            preview = ", ".join(missing_middle[:10])
            raise ValueError(
                f"Missing Best_Prompt_Slice_ID for {len(missing_middle)} train cases. "
                f"Examples: {preview}. Please complete prompt table or use --allow-missing-middle"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdir = sample["pdir"]
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.image_name)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.mask_name)))
        gt = (gt > 0).astype(np.uint8)

        frames = []
        h0, w0 = img.shape[1], img.shape[2]
        for local_t in range(img.shape[0]):
            u8 = window_to_uint8(img[local_t], self.window_center, self.window_width)
            rgb = np.stack([u8, u8, u8], axis=0)  # [3, H, W]
            image_tensor = torch.from_numpy(rgb).float() / 255.0
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_tensor = torch.from_numpy(gt[local_t]).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(
                mask_tensor,
                size=(self.input_size, self.input_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.bool)

            frames.append(
                Frame(
                    data=image_tensor,
                    objects=[Object(object_id=1, frame_index=local_t, segment=mask_tensor)],
                )
            )

        return VideoDatapoint(
            frames=frames,
            video_id=int(sample["video_id"]),
            size=(h0, w0),
        )


def load_model_cfg_dict_once(model_cfg: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=model_cfg)
    return OmegaConf.to_container(cfg.model, resolve=True)


def build_model(
    model_cfg_dict_template,
    pretrained_ckpt: Path,
    freeze_image_encoder: bool,
    middle_prompt_by_video_id,
    device: torch.device,
):
    model_cfg_dict = dict(model_cfg_dict_template)
    model_cfg_dict["freeze_image_encoder"] = freeze_image_encoder

    image_encoder_cfg = model_cfg_dict.pop("image_encoder")
    memory_attention_cfg = model_cfg_dict.pop("memory_attention")
    memory_encoder_cfg = model_cfg_dict.pop("memory_encoder")
    model_cfg_dict.pop("_target_", None)

    image_encoder = instantiate(image_encoder_cfg, _recursive_=True)
    memory_attention = instantiate(memory_attention_cfg, _recursive_=True)
    memory_encoder = instantiate(memory_encoder_cfg, _recursive_=True)

    model = SAM2TrainUpperLowerMiddleMask(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        middle_prompt_by_video_id=middle_prompt_by_video_id,
        **model_cfg_dict,
    )

    state = torch.load(str(pretrained_ckpt), map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys while loading pretrained: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[WARN] missing keys while loading pretrained: {len(missing)}")

    model = model.to(device)
    return model


def build_optimizer(model, base_lr: float, vision_lr: float, weight_decay: float):
    image_encoder_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("image_encoder"):
            image_encoder_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if len(other_params) > 0:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if len(image_encoder_params) > 0:
        groups.append({"params": image_encoder_params, "lr": vision_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(groups)


def compute_batch_volume_dice(outputs, batch_masks: torch.Tensor) -> float:
    # outputs: list[T] each has pred_masks_high_res [O,1,H,W], batch_masks: [T,O,H,W]
    t_dim, o_dim = batch_masks.shape[:2]
    dices = []
    for o in range(o_dim):
        pred_stack = []
        gt_stack = []
        for t in range(t_dim):
            pred = outputs[t]["pred_masks_high_res"][o, 0]
            gt = batch_masks[t, o].float()
            pred_stack.append((pred > 0).float())
            gt_stack.append(gt)
        pred_vol = torch.stack(pred_stack, dim=0)
        gt_vol = torch.stack(gt_stack, dim=0)
        inter = (pred_vol * gt_vol).sum()
        denom = pred_vol.sum() + gt_vol.sum()
        dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        dices.append(float(dice.item()))
    return float(np.mean(dices)) if len(dices) else 0.0


def ddp_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def reduce_scalar(value: float, device: torch.device) -> float:
    if not ddp_enabled():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"[WARN] destroy_process_group failed: {e}")


def configure_cuda_allocator(
    max_split_size_mb: int,
    gc_threshold: float,
    expandable_segments: bool,
):
    conf_items = []
    if max_split_size_mb > 0:
        conf_items.append(f"max_split_size_mb:{int(max_split_size_mb)}")
    if gc_threshold > 0:
        conf_items.append(f"garbage_collection_threshold:{float(gc_threshold)}")
    conf_items.append(f"expandable_segments:{'True' if expandable_segments else 'False'}")
    conf = ",".join(conf_items)
    prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"{prev},{conf}" if prev else conf
    print(f"[INFO] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")


def _clone_backbone_out(backbone_out: dict) -> dict:
    # Shallow copy is enough: tensors are immutable for our usage here.
    return {k: v for k, v in backbone_out.items()}


def _precompute_backbone_out(core_model, batch, forward_backbone_per_frame: bool = False):
    if forward_backbone_per_frame:
        return {"backbone_fpn": None, "vision_pos_enc": None}
    if core_model.training or not core_model.forward_backbone_per_frame_for_eval:
        return core_model.forward_image(batch.flat_img_batch)
    return {"backbone_fpn": None, "vision_pos_enc": None}


def _outputs_from_tracking_dict(output_dict: dict, num_frames: int):
    all_frame_outputs = {}
    all_frame_outputs.update(output_dict["cond_frame_outputs"])
    all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
    all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
    # Keep behavior aligned with SAM2Train.forward_tracking.
    all_frame_outputs = [
        {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
    ]
    return all_frame_outputs


def _forward_tracking_iterative(
    core_model,
    backbone_out,
    batch,
    output_dict=None,
    processing_order=None,
):
    img_feats_already_computed = backbone_out["backbone_fpn"] is not None
    if img_feats_already_computed:
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = core_model._prepare_backbone_features(backbone_out)

    num_frames = backbone_out["num_frames"]
    init_cond_frames = backbone_out["init_cond_frames"]
    frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]

    if output_dict is None:
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
    if processing_order is None:
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

    for stage_id in processing_order:
        img_ids = batch.flat_obj_to_img_idx[stage_id]
        if img_feats_already_computed:
            current_vision_feats = [x[:, img_ids] for x in vision_feats]
            current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
        else:
            (
                _,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
            ) = core_model._prepare_backbone_features_per_frame(
                batch.flat_img_batch, img_ids
            )

        current_out = core_model.track_step(
            frame_idx=stage_id,
            is_init_cond_frame=stage_id in init_cond_frames,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
            mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
            gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
        )
        add_output_as_cond_frame = stage_id in init_cond_frames or (
            core_model.add_all_frames_to_correct_as_cond
            and stage_id in frames_to_add_correction_pt
        )
        if add_output_as_cond_frame:
            output_dict["cond_frame_outputs"][stage_id] = current_out
            output_dict["non_cond_frame_outputs"].pop(stage_id, None)
        else:
            output_dict["non_cond_frame_outputs"][stage_id] = current_out
            output_dict["cond_frame_outputs"].pop(stage_id, None)

    return output_dict


def _append_middle_prompts_inplace(core_model, base_backbone_out, backbone_stage1, batch):
    # Build stage-2 prompt layout, then append only the new middle prompts to stage-1 backbone_out.
    stage2_backbone = core_model.prepare_prompt_inputs(
        _clone_backbone_out(base_backbone_out), batch
    )
    old_init = set(backbone_stage1["init_cond_frames"])
    new_init = set(stage2_backbone["init_cond_frames"])
    middle_frames = sorted(new_init - old_init)

    for t in middle_frames:
        if t in stage2_backbone["mask_inputs_per_frame"]:
            backbone_stage1["mask_inputs_per_frame"][t] = stage2_backbone["mask_inputs_per_frame"][t]

    merged_init = sorted(old_init | set(middle_frames))
    backbone_stage1["init_cond_frames"] = merged_init
    num_frames = int(backbone_stage1["num_frames"])
    backbone_stage1["frames_not_in_init_cond"] = [t for t in range(num_frames) if t not in set(merged_init)]
    return backbone_stage1, middle_frames


def run_epoch_two_pass(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    amp_dtype,
    train_mode: bool,
    stage1_loss_weight: float,
    stage2_loss_weight: float,
    two_pass_mode: str,
    forward_backbone_per_frame: bool = False,
    empty_cache_every: int = 0,
):
    model.train(train_mode)
    total_loss = 0.0
    total_dice = 0.0
    n_batch = 0

    core_model = unwrap_model(model)

    for batch_idx, batch in enumerate(loader, start=1):
        if train_mode:
            # Free previous step gradients before forward to reduce peak memory.
            optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device, non_blocking=True)
        base_backbone_out = _precompute_backbone_out(
            core_model,
            batch,
            forward_backbone_per_frame=forward_backbone_per_frame,
        )

        # Pass-1: boundary prompts only
        core_model.set_middle_prompt_enabled(False)
        backbone_stage1 = core_model.prepare_prompt_inputs(
            _clone_backbone_out(base_backbone_out), batch
        )
        stage1_needs_grad = stage1_loss_weight > 0.0
        grad_ctx = nullcontext() if stage1_needs_grad else torch.no_grad()
        with grad_ctx:
            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda"),
                dtype=amp_dtype,
            ):
                output_dict_stage1 = _forward_tracking_iterative(
                    core_model, backbone_stage1, batch
                )
                if stage1_needs_grad:
                    outputs_stage1 = _outputs_from_tracking_dict(
                        output_dict_stage1, backbone_stage1["num_frames"]
                    )
                    loss_dict_stage1 = loss_fn(outputs_stage1, batch.masks)
                    loss_stage1 = (
                        loss_dict_stage1["core_loss"]
                        if isinstance(loss_dict_stage1, dict)
                        else loss_dict_stage1
                    )
                else:
                    loss_stage1 = torch.zeros((), device=device, dtype=torch.float32)

        # Pass-2:
        # - iterative: continue on stage-1 memory and inject middle prompt
        # - independent: fresh second forward with upper/lower/middle
        core_model.set_middle_prompt_enabled(True)
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda"),
            dtype=amp_dtype,
        ):
            if two_pass_mode == "iterative":
                backbone_stage2, middle_frames = _append_middle_prompts_inplace(
                    core_model, base_backbone_out, backbone_stage1, batch
                )
                output_dict_stage2 = _forward_tracking_iterative(
                    core_model,
                    backbone_stage2,
                    batch,
                    output_dict=output_dict_stage1,
                    processing_order=middle_frames + [
                        t
                        for t in backbone_stage2["frames_not_in_init_cond"]
                        if t not in set(middle_frames)
                    ],
                )
            else:
                backbone_stage2 = core_model.prepare_prompt_inputs(
                    _clone_backbone_out(base_backbone_out), batch
                )
                output_dict_stage2 = _forward_tracking_iterative(
                    core_model,
                    backbone_stage2,
                    batch,
                )
            outputs_stage2 = _outputs_from_tracking_dict(
                output_dict_stage2, backbone_stage2["num_frames"]
            )
            loss_dict_stage2 = loss_fn(outputs_stage2, batch.masks)
            loss_stage2 = (
                loss_dict_stage2["core_loss"]
                if isinstance(loss_dict_stage2, dict)
                else loss_dict_stage2
            )
            loss = stage1_loss_weight * loss_stage1 + stage2_loss_weight * loss_stage2

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_dice += compute_batch_volume_dice(outputs_stage2, batch.masks)
        n_batch += 1

        core_model.set_middle_prompt_enabled(False)
        # Release large per-batch references as early as possible.
        del outputs_stage2, output_dict_stage2, backbone_stage2
        del backbone_stage1, output_dict_stage1, base_backbone_out
        del loss_stage2, loss_stage1, loss
        if (
            empty_cache_every > 0
            and device.type == "cuda"
            and (batch_idx % empty_cache_every) == 0
        ):
            torch.cuda.empty_cache()

    if n_batch == 0:
        avg_loss, avg_dice = 0.0, 0.0
    else:
        avg_loss, avg_dice = total_loss / n_batch, total_dice / n_batch
    avg_loss = reduce_scalar(avg_loss, device)
    avg_dice = reduce_scalar(avg_dice, device)
    return avg_loss, avg_dice


def is_main_process() -> bool:
    if not ddp_enabled():
        return True
    return dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def save_ckpt(path: Path, epoch: int, model, optimizer, scheduler, scaler, best_val_dice: float):
    if not is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_dice": best_val_dice,
        },
        str(path),
    )


def load_ckpt(path: Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    state = torch.load(str(path), map_location=map_location)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    unwrap_model(model).load_state_dict(model_state, strict=False)

    if optimizer is not None and isinstance(state, dict) and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and isinstance(state, dict) and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and isinstance(state, dict) and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    last_epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
    best_val_dice = float(state.get("best_val_dice", -1.0)) if isinstance(state, dict) else -1.0
    return last_epoch, best_val_dice


def make_folds(patient_dirs, num_folds: int, seed: int):
    patient_dirs = list(patient_dirs)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(patient_dirs))
    rng.shuffle(idx)
    folds = np.array_split(idx, num_folds)
    out = []
    for fold_idx in range(num_folds):
        val_idx = set(folds[fold_idx].tolist())
        train = [patient_dirs[i] for i in range(len(patient_dirs)) if i not in val_idx]
        val = [patient_dirs[i] for i in range(len(patient_dirs)) if i in val_idx]
        out.append((train, val))
    return out


@record
def main():
    parser = argparse.ArgumentParser("SAM2 upper/lower->middle iterative-mask prompt finetuning")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT, help="Directory containing train patient folders")
    parser.add_argument("--prompt3-xlsx", type=Path, default=DEFAULT_PROMPT3_XLSX, help="mask_prompt_3.py output excel (contains Best_Prompt_Slice_ID)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for folds/checkpoints/logs")
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=DEFAULT_MODEL_CFG,
        help="SAM2 model config in Hydra (e.g. configs/sam2.1/sam2.1_hiera_l.yaml)",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=Path,
        default=DEFAULT_PRETRAINED_CKPT,
        help="Pretrained checkpoint for finetuning",
    )
    parser.add_argument("--image-name", type=str, default="image.nii.gz")
    parser.add_argument("--mask-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--vision-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eta-min-factor", type=float, default=0.1)
    parser.add_argument("--freeze-image-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-image-encoder", dest="freeze_image_encoder", action="store_false")
    parser.add_argument("--allow-missing-middle", action="store_true", help="Allow cases missing Best_Prompt_Slice_ID and fallback in model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--forward-backbone-per-frame",
        action="store_true",
        help="Compute image backbone features on demand per frame (lower GPU memory, slower).",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N batches (0 disables). Useful for fragmentation OOM.",
    )
    parser.add_argument(
        "--cuda-alloc-max-split-mb",
        type=int,
        default=128,
        help="Set max_split_size_mb in PYTORCH_CUDA_ALLOC_CONF (0 disables).",
    )
    parser.add_argument(
        "--cuda-alloc-gc-threshold",
        type=float,
        default=0.8,
        help="Set garbage_collection_threshold in PYTORCH_CUDA_ALLOC_CONF (<=0 disables).",
    )
    parser.add_argument(
        "--cuda-alloc-expandable-segments",
        action="store_true",
        default=True,
        help="Enable expandable_segments in PYTORCH_CUDA_ALLOC_CONF.",
    )
    parser.add_argument(
        "--no-cuda-alloc-expandable-segments",
        dest="cuda_alloc_expandable_segments",
        action="store_false",
    )
    parser.add_argument("--stage1-loss-weight", type=float, default=0.0)
    parser.add_argument("--stage2-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--two-pass-mode",
        type=str,
        default="iterative",
        choices=["iterative", "independent"],
        help="iterative: stage2 continues on stage1 memory; independent: two standalone forwards",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each fold from fold_x/checkpoints/last.pth when available.",
    )
    parser.add_argument(
        "--resume-folds",
        type=str,
        default="",
        help="Optional comma-separated fold ids to resume only, e.g. '2,3'. Empty means all folds.",
    )
    args = parser.parse_args()
    configure_cuda_allocator(
        max_split_size_mb=args.cuda_alloc_max_split_mb,
        gc_threshold=args.cuda_alloc_gc_threshold,
        expandable_segments=args.cuda_alloc_expandable_segments,
    )

    # Full-case training uses variable number of frames across patients.
    # training.utils.data_utils.collate_fn requires equal T within one batch.
    # Therefore batch size must be 1 unless we implement temporal padding.
    if args.batch_size != 1:
        print(
            f"[WARN] batch_size={args.batch_size} is not supported for full-case variable-length videos. "
            "Force set to 1."
        )
        args.batch_size = 1

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = 0

    if not args.train_root.exists():
        raise FileNotFoundError(f"train root not found: {args.train_root}")
    if not args.pretrained_ckpt.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrained_ckpt}")

    set_seed(args.seed)
    if use_ddp:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    middle_prompt_by_video_id = load_middle_prompt_map(args.prompt3_xlsx)

    patient_dirs = sorted([p for p in args.train_root.iterdir() if p.is_dir()], key=patient_sort_key)
    if len(patient_dirs) < args.num_folds:
        raise ValueError(f"patients ({len(patient_dirs)}) < num_folds ({args.num_folds})")

    if is_main_process():
        args.output_root.mkdir(parents=True, exist_ok=True)
    split_csv = args.output_root / "fold_split.csv"

    fold_defs = make_folds(patient_dirs, args.num_folds, args.seed)
    if is_main_process():
        with open(split_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "split", "patient"])
            for i, (train_list, val_list) in enumerate(fold_defs):
                for p in train_list:
                    writer.writerow([i, "train", p.name])
                for p in val_list:
                    writer.writerow([i, "val", p.name])

    model_cfg_dict_template = load_model_cfg_dict_once(args.model_cfg)

    summary_rows = []
    resume_fold_set = None
    if args.resume_folds.strip():
        resume_fold_set = set(int(x.strip()) for x in args.resume_folds.split(",") if x.strip())
    for fold_idx, (train_patients, val_patients) in enumerate(fold_defs):
        if is_main_process():
            print(f"\n[Fold {fold_idx}] train={len(train_patients)} val={len(val_patients)}")
        fold_dir = args.output_root / f"fold_{fold_idx}"
        ckpt_dir = fold_dir / "checkpoints"
        log_dir = fold_dir / "logs"
        if is_main_process():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

        train_ds = UpperLowerMiddleVolumeDataset(
            train_patients,
            middle_prompt_by_video_id=middle_prompt_by_video_id,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
            strict_middle=(not args.allow_missing_middle),
        )
        val_ds = UpperLowerMiddleVolumeDataset(
            val_patients,
            middle_prompt_by_video_id=middle_prompt_by_video_id,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
            strict_middle=(not args.allow_missing_middle),
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, dict_key="all"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            sampler=val_sampler,
            collate_fn=lambda b: collate_fn(b, dict_key="all"),
        )

        model = build_model(
            model_cfg_dict_template=model_cfg_dict_template,
            pretrained_ckpt=args.pretrained_ckpt,
            freeze_image_encoder=args.freeze_image_encoder,
            middle_prompt_by_video_id=middle_prompt_by_video_id,
            device=device,
        )
        if use_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        optimizer = build_optimizer(model, args.base_lr, args.vision_lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=args.base_lr * args.eta_min_factor,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
        loss_fn = MultiStepMultiMasksAndIous(
            weight_dict={"loss_mask": 20, "loss_dice": 1, "loss_iou": 1, "loss_class": 1},
            supervise_all_iou=True,
            iou_use_l1_loss=True,
            pred_obj_scores=True,
            focal_gamma_obj_score=0.0,
            focal_alpha_obj_score=-1.0,
        ).to(device)

        best_val_dice = -1.0
        best_epoch = -1
        history = []
        start_epoch = 0
        if args.resume and (resume_fold_set is None or fold_idx in resume_fold_set):
            resume_ckpt = ckpt_dir / "last.pth"
            if resume_ckpt.exists():
                start_epoch, best_val_dice = load_ckpt(
                    path=resume_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    map_location="cpu",
                )
                best_epoch = start_epoch
                if is_main_process():
                    print(
                        f"[Fold {fold_idx}] Resume from {resume_ckpt} | "
                        f"last_epoch={start_epoch} -> next_epoch={start_epoch + 1}, "
                        f"best_val_dice={best_val_dice:.6f}"
                    )
                hist_json = log_dir / "history.json"
                if hist_json.exists():
                    try:
                        with open(hist_json, "r", encoding="utf-8") as f:
                            old_history = json.load(f)
                        if isinstance(old_history, list):
                            history = old_history
                    except Exception:
                        if is_main_process():
                            print(f"[Fold {fold_idx}] WARN: failed to read history.json, continue with empty history")

        if start_epoch >= args.epochs:
            if is_main_process():
                print(
                    f"[Fold {fold_idx}] Already finished "
                    f"(last_epoch={start_epoch}, target_epochs={args.epochs}), skip training loop."
                )

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            tr_loss, tr_dice = run_epoch_two_pass(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                train_mode=True,
                stage1_loss_weight=args.stage1_loss_weight,
                stage2_loss_weight=args.stage2_loss_weight,
                two_pass_mode=args.two_pass_mode,
                forward_backbone_per_frame=args.forward_backbone_per_frame,
                empty_cache_every=args.empty_cache_every,
            )
            with torch.no_grad():
                va_loss, va_dice = run_epoch_two_pass(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    amp_dtype=amp_dtype,
                    train_mode=False,
                    stage1_loss_weight=args.stage1_loss_weight,
                    stage2_loss_weight=args.stage2_loss_weight,
                    two_pass_mode=args.two_pass_mode,
                    forward_backbone_per_frame=args.forward_backbone_per_frame,
                    empty_cache_every=args.empty_cache_every,
                )
            scheduler.step(va_dice)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": tr_loss,
                    "train_dice": tr_dice,
                    "val_loss": va_loss,
                    "val_dice": va_dice,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if is_main_process():
                print(
                    f"[Fold {fold_idx}] Epoch {epoch+1}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | "
                    f"val_loss={va_loss:.4f} val_dice={va_dice:.4f}"
                )

            save_ckpt(
                ckpt_dir / "last.pth",
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val_dice,
            )
            if va_dice > best_val_dice:
                best_val_dice = va_dice
                best_epoch = epoch + 1
                save_ckpt(
                    ckpt_dir / "best.pth",
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_val_dice,
                )

        if is_main_process():
            with open(log_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        if is_main_process():
            summary_rows.append(
                {
                    "fold": fold_idx,
                    "best_val_dice": best_val_dice,
                    "best_epoch": best_epoch,
                    "best_ckpt": str((ckpt_dir / "best.pth").resolve()),
                    "last_ckpt": str((ckpt_dir / "last.pth").resolve()),
                    "num_train_cases": len(train_ds),
                    "num_val_cases": len(val_ds),
                }
            )

    if is_main_process():
        summary_csv = args.output_root / "cv_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fold",
                    "best_val_dice",
                    "best_epoch",
                    "best_ckpt",
                    "last_ckpt",
                    "num_train_cases",
                    "num_val_cases",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        best_fold = max(summary_rows, key=lambda x: x["best_val_dice"])
        with open(args.output_root / "best_fold.txt", "w", encoding="utf-8") as f:
            f.write(f"best_fold: {best_fold['fold']}\n")
            f.write(f"best_val_dice: {best_fold['best_val_dice']:.6f}\n")
            f.write(f"best_epoch: {best_fold['best_epoch']}\n")
            f.write(f"best_ckpt: {best_fold['best_ckpt']}\n")

        print("\n[DONE] Training finished.")
        print(f"[DONE] CV summary: {summary_csv}")
        print(f"[DONE] Best fold: {best_fold['fold']} | val_dice={best_fold['best_val_dice']:.4f}")

    _cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup_ddp()
