#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

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
    raise RuntimeError("Cannot locate SAM2 project root. Set SAM2_PROJECT_ROOT if needed.")


PROJECT_ROOT = _find_project_root(CURRENT_DIR)
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model.sam2 import SAM2Train
from training.utils.data_utils import Frame, Object, VideoDatapoint, collate_fn


DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


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


def _sample_aug_params(enable_augment: bool):
    if not enable_augment:
        return None
    return {
        "hflip": random.random() < 0.5,
        "vflip": random.random() < 0.3,
        "rot_k": random.randint(0, 3),
        "contrast": random.uniform(0.9, 1.1),
        "brightness": random.uniform(-0.05, 0.05),
        "noise_std": random.uniform(0.0, 0.02),
    }


def _apply_aug_to_slice(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, aug_params: dict):
    if aug_params is None:
        return image_tensor, mask_tensor
    if aug_params["hflip"]:
        image_tensor = torch.flip(image_tensor, dims=[2])
        mask_tensor = torch.flip(mask_tensor, dims=[1])
    if aug_params["vflip"]:
        image_tensor = torch.flip(image_tensor, dims=[1])
        mask_tensor = torch.flip(mask_tensor, dims=[0])
    if aug_params["rot_k"] > 0:
        image_tensor = torch.rot90(image_tensor, aug_params["rot_k"], dims=[1, 2])
        mask_tensor = torch.rot90(mask_tensor, aug_params["rot_k"], dims=[0, 1])
    image_tensor = image_tensor * aug_params["contrast"] + aug_params["brightness"]
    if aug_params["noise_std"] > 0:
        image_tensor = image_tensor + torch.randn_like(image_tensor) * aug_params["noise_std"]
    image_tensor = image_tensor.clamp(0.0, 1.0)
    return image_tensor, mask_tensor


def patient_sort_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


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


def normalize_patient_token(value) -> str:
    s = str(value).strip()
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    if m is None:
        return ""
    return f"CTV_{int(m.group(1)):03d}"


def parse_prompt_slices(text) -> list[int]:
    s = str(text).strip()
    if not s:
        return []
    return [int(x) for x in re.findall(r"-?\d+", s)]


def parse_k_list(k_text: str) -> list[int]:
    out = []
    for seg in str(k_text).split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" in seg:
            a, b = seg.split("-", 1)
            lo, hi = min(int(a.strip()), int(b.strip())), max(int(a.strip()), int(b.strip()))
            out.extend(list(range(lo, hi + 1)))
        else:
            out.append(int(seg))
    out = sorted(set(out))
    if len(out) == 0:
        raise ValueError("Empty K list")
    return out


def load_prompt_map_for_k(prompt_xlsx: Path, k: int) -> dict[int, list[int]]:
    if not prompt_xlsx.exists():
        raise FileNotFoundError(f"Prompt table not found: {prompt_xlsx}")
    sheets = pd.read_excel(prompt_xlsx, sheet_name=None)
    sheet_name = f"K{k}"
    if sheet_name not in sheets:
        raise ValueError(f"Missing sheet {sheet_name} in {prompt_xlsx}")

    df = sheets[sheet_name].copy()
    cols = {c.lower(): c for c in df.columns}
    if "patientid" not in cols or "promptslices" not in cols:
        raise ValueError(f"Sheet {sheet_name} requires PatientID and PromptSlices columns")

    pid_col = cols["patientid"]
    slice_col = cols["promptslices"]
    mapping = {}
    for _, row in df.iterrows():
        pid = normalize_patient_token(row.get(pid_col, ""))
        if not pid:
            continue
        vid = patient_video_num_from_id(pid)
        slices = parse_prompt_slices(row.get(slice_col, ""))
        if len(slices) == 0:
            continue
        mapping[vid] = list(dict.fromkeys(int(x) for x in slices))

    if len(mapping) == 0:
        raise ValueError(f"No valid prompt entries in {sheet_name}")
    return mapping


class SAM2TrainPromptListMask(SAM2Train):
    def __init__(self, *args, prompt_slices_by_video_id=None, **kwargs):
        kwargs.update(
            dict(
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
            )
        )
        super().__init__(*args, **kwargs)
        self.prompt_slices_by_video_id = {
            int(k): [int(x) for x in v] for k, v in (prompt_slices_by_video_id or {}).items()
        }

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        gt_masks_per_frame = {frame_idx: masks.unsqueeze(1) for frame_idx, masks in enumerate(input.masks)}
        num_frames = input.num_frames

        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["num_frames"] = num_frames
        backbone_out["use_pt_input"] = False
        backbone_out["point_inputs_per_frame"] = {}
        backbone_out["frames_to_add_correction_pt"] = []

        masks_tohw = input.masks
        t_dim, o_dim = masks_tohw.shape[:2]
        obj_video_ids = input.metadata.unique_objects_identifier[0, :, 0].to(torch.long)

        prompts_per_object = []
        init_cond_frames = set()
        for obj_idx in range(o_dim):
            gt_obj = masks_tohw[:, obj_idx]
            pos_t = torch.nonzero(gt_obj.flatten(1).any(dim=1), as_tuple=False).flatten().tolist()
            pos_t = [int(x) for x in pos_t]

            if len(pos_t) == 0:
                prompts = [int(start_frame_idx)]
            else:
                lower = int(min(pos_t))
                upper = int(max(pos_t))
                vid = int(obj_video_ids[obj_idx].item())
                external = self.prompt_slices_by_video_id.get(vid, [])
                valid = []
                for sid in external:
                    sid = int(max(0, min(int(sid), t_dim - 1)))
                    if bool(gt_obj[sid].any()):
                        valid.append(sid)
                valid = list(dict.fromkeys(valid))
                if len(valid) == 0:
                    valid = [lower] if lower == upper else [lower, upper]
                prompts = valid

            prompts_per_object.append(prompts)
            init_cond_frames.update(prompts)

        init_cond_frames = sorted(int(x) for x in init_cond_frames)
        init_set = set(init_cond_frames)
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [t for t in range(start_frame_idx, num_frames) if t not in init_set]

        backbone_out["mask_inputs_per_frame"] = {}
        for t in init_cond_frames:
            gt_t = gt_masks_per_frame[t]
            prompt_t = torch.zeros_like(gt_t)
            for o in range(o_dim):
                if t in prompts_per_object[o]:
                    prompt_t[o] = gt_t[o]
            backbone_out["mask_inputs_per_frame"][t] = prompt_t

        return backbone_out

class VolumeDataset(Dataset):
    def __init__(
        self,
        patient_dirs,
        image_name="image.nii.gz",
        mask_name="CTV.nii.gz",
        window_center=40.0,
        window_width=400.0,
        input_size=1024,
        enable_augment=False,
    ):
        self.patient_dirs = list(patient_dirs)
        self.image_name = image_name
        self.mask_name = mask_name
        self.window_center = float(window_center)
        self.window_width = float(window_width)
        self.input_size = int(input_size)
        self.enable_augment = bool(enable_augment)

        self.samples = []
        for pdir in self.patient_dirs:
            img_path = pdir / self.image_name
            gt_path = pdir / self.mask_name
            if not img_path.exists() or not gt_path.exists():
                continue
            gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
            gt = (gt > 0).astype(np.uint8)
            pos = np.where(gt.sum(axis=(1, 2)) > 0)[0]
            if len(pos) == 0:
                continue
            patient_id = patient_id_from_folder(pdir)
            video_id = patient_video_num_from_id(patient_id)
            self.samples.append({"pdir": pdir, "video_id": video_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdir = sample["pdir"]
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.image_name)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.mask_name)))
        gt = (gt > 0).astype(np.uint8)
        aug_params = _sample_aug_params(self.enable_augment)

        frames = []
        h0, w0 = img.shape[1], img.shape[2]
        for local_t in range(img.shape[0]):
            u8 = window_to_uint8(img[local_t], self.window_center, self.window_width)
            rgb = np.stack([u8, u8, u8], axis=0)
            image_tensor = torch.from_numpy(rgb).float() / 255.0
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_tensor = torch.from_numpy(gt[local_t]).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, size=(self.input_size, self.input_size), mode="nearest")
            mask_tensor = mask_tensor.squeeze(0).squeeze(0).to(torch.bool)
            image_tensor, mask_tensor = _apply_aug_to_slice(image_tensor, mask_tensor, aug_params)

            frames.append(
                Frame(data=image_tensor, objects=[Object(object_id=1, frame_index=local_t, segment=mask_tensor)])
            )

        return VideoDatapoint(frames=frames, video_id=int(sample["video_id"]), size=(h0, w0))


def load_model_cfg_dict_once(model_cfg: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=model_cfg)
    return OmegaConf.to_container(cfg.model, resolve=True)


def build_model(model_cfg_dict_template, init_ckpt: Path, freeze_image_encoder: bool, prompt_slices_by_video_id, device):
    model_cfg_dict = dict(model_cfg_dict_template)
    model_cfg_dict["freeze_image_encoder"] = freeze_image_encoder

    image_encoder_cfg = model_cfg_dict.pop("image_encoder")
    memory_attention_cfg = model_cfg_dict.pop("memory_attention")
    memory_encoder_cfg = model_cfg_dict.pop("memory_encoder")
    model_cfg_dict.pop("_target_", None)

    image_encoder = instantiate(image_encoder_cfg, _recursive_=True)
    memory_attention = instantiate(memory_attention_cfg, _recursive_=True)
    memory_encoder = instantiate(memory_encoder_cfg, _recursive_=True)

    model = SAM2TrainPromptListMask(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        prompt_slices_by_video_id=prompt_slices_by_video_id,
        **model_cfg_dict,
    )

    state = torch.load(str(init_ckpt), map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys while loading pretrained: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[WARN] missing keys while loading pretrained: {len(missing)}")
    return model.to(device)


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


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + 1e-5) / (probs.sum() + targets.sum() + 1e-5)
        dice_loss = 1 - dice
        return 0.5 * bce + 0.5 * dice_loss


def compute_dice_bce_loss(outputs, batch_masks: torch.Tensor, criterion: DiceBCELoss):
    losses = []
    for t in range(batch_masks.shape[0]):
        logits = outputs[t]["pred_masks_high_res"][:, 0]
        target = batch_masks[t].float()
        losses.append(criterion(logits, target))
    if len(losses) == 0:
        return torch.zeros((), device=batch_masks.device, dtype=torch.float32)
    return torch.stack(losses, dim=0).mean()


def compute_batch_volume_dice(outputs, batch_masks: torch.Tensor) -> float:
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
        dices.append(float(((2.0 * inter + 1e-6) / (denom + 1e-6)).item()))
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


def run_epoch(model, loader, optimizer, scaler, device, amp_dtype, train_mode: bool):
    model.train(train_mode)
    total_loss = 0.0
    total_dice = 0.0
    n_batch = 0
    criterion = DiceBCELoss().to(device)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
            outputs = model(batch)
            loss = compute_dice_bce_loss(outputs, batch.masks, criterion)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_dice += compute_batch_volume_dice(outputs, batch.masks)
        n_batch += 1

    if n_batch == 0:
        avg_loss, avg_dice = 0.0, 0.0
    else:
        avg_loss, avg_dice = total_loss / n_batch, total_dice / n_batch
    return reduce_scalar(avg_loss, device), reduce_scalar(avg_dice, device)


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
    return int(state.get("epoch", 0)), float(state.get("best_val_dice", -1.0))

def make_folds(patient_dirs, num_folds: int, seed: int):
    patient_dirs = list(patient_dirs)
    if num_folds <= 1:
        rng = np.random.RandomState(seed)
        idx = np.arange(len(patient_dirs))
        rng.shuffle(idx)
        val_count = max(1, int(round(len(patient_dirs) * 0.2)))
        if val_count >= len(patient_dirs):
            val_count = max(1, len(patient_dirs) - 1)
        val_idx = set(idx[:val_count].tolist())
        train = [patient_dirs[i] for i in range(len(patient_dirs)) if i not in val_idx]
        val = [patient_dirs[i] for i in range(len(patient_dirs)) if i in val_idx]
        return [(train, val)]

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


def train_for_single_k(
    args,
    k,
    prompt_map_k,
    init_ckpt,
    model_cfg_dict_template,
    device,
    amp_dtype,
    use_ddp,
    local_rank,
    fold_defs,
):
    if is_main_process():
        print(f"\n========== [K={k}] ==========")

    k_root = args.output_root / f"K{k}"
    if is_main_process():
        k_root.mkdir(parents=True, exist_ok=True)

    if is_main_process():
        with open(k_root / "fold_split.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "split", "patient"])
            for i, (train_list, val_list) in enumerate(fold_defs):
                for p in train_list:
                    writer.writerow([i, "train", p.name])
                for p in val_list:
                    writer.writerow([i, "val", p.name])

    summary_rows = []
    resume_fold_set = None
    if args.resume_folds.strip():
        resume_fold_set = set(int(x.strip()) for x in args.resume_folds.split(",") if x.strip())

    for fold_idx, (train_patients, val_patients) in enumerate(fold_defs):
        fold_dir = k_root / f"fold_{fold_idx}"
        ckpt_dir = fold_dir / "checkpoints"
        log_dir = fold_dir / "logs"
        if is_main_process():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[K={k} Fold {fold_idx}] train={len(train_patients)} val={len(val_patients)}")

        train_ds = VolumeDataset(
            train_patients,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
            enable_augment=True,
        )
        val_ds = VolumeDataset(
            val_patients,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
            enable_augment=False,
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
            init_ckpt=init_ckpt,
            freeze_image_encoder=args.freeze_image_encoder,
            prompt_slices_by_video_id=prompt_map_k,
            device=device,
        )
        if use_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        optimizer = build_optimizer(model, args.base_lr, args.vision_lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            threshold=1e-3,
            cooldown=1,
            min_lr=1e-6,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

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
                hist_json = log_dir / "history.json"
                if hist_json.exists():
                    try:
                        history = json.loads(hist_json.read_text(encoding="utf-8"))
                    except Exception:
                        history = []

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            tr_loss, tr_dice = run_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, True)
            with torch.no_grad():
                va_loss, va_dice = run_epoch(model, val_loader, optimizer, scaler, device, amp_dtype, False)
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
                    f"[K={k} Fold {fold_idx}] Epoch {epoch + 1}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} | "
                    f"val_loss={va_loss:.4f} val_dice={va_dice:.4f}"
                )

            save_ckpt(ckpt_dir / "last.pth", epoch + 1, model, optimizer, scheduler, scaler, best_val_dice)
            if va_dice > best_val_dice:
                best_val_dice = va_dice
                best_epoch = epoch + 1
                save_ckpt(ckpt_dir / "best.pth", epoch + 1, model, optimizer, scheduler, scaler, best_val_dice)

        if is_main_process():
            with open(log_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            summary_rows.append(
                {
                    "k": k,
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
        cv_csv = k_root / "cv_summary.csv"
        with open(cv_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "k",
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
        with open(k_root / "best_fold.txt", "w", encoding="utf-8") as f:
            f.write(f"k: {k}\n")
            f.write(f"best_fold: {best_fold['fold']}\n")
            f.write(f"best_val_dice: {best_fold['best_val_dice']:.6f}\n")
            f.write(f"best_epoch: {best_fold['best_epoch']}\n")
            f.write(f"best_ckpt: {best_fold['best_ckpt']}\n")

        print(f"[K={k}] done. best_fold={best_fold['fold']} best_val_dice={best_fold['best_val_dice']:.4f}")
        return {
            "k": k,
            "best_fold": best_fold["fold"],
            "best_val_dice": best_fold["best_val_dice"],
            "best_ckpt": best_fold["best_ckpt"],
        }
    return None


def main():
    parser = argparse.ArgumentParser("SAM2 one-shot multi-prompt K2..K10 finetuning")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-xlsx", type=Path, required=True)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--pretrained-ckpt", type=Path, required=True)
    parser.add_argument("--ks", type=str, default="2-10")
    parser.add_argument("--image-name", type=str, default="image.nii.gz")
    parser.add_argument("--mask-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-folds", type=str, default="")
    args = parser.parse_args()

    if args.batch_size != 1:
        print(f"[WARN] batch_size={args.batch_size} is not supported for variable-length videos. Force to 1.")
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

    init_ckpt = args.pretrained_ckpt
    set_seed(args.seed)

    if use_ddp:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    if is_main_process():
        args.output_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] pretrained checkpoint: {init_ckpt}")

    patient_dirs = sorted([p for p in args.train_root.iterdir() if p.is_dir()], key=patient_sort_key)
    if len(patient_dirs) < args.num_folds:
        raise ValueError(f"patients ({len(patient_dirs)}) < num_folds ({args.num_folds})")
    fold_defs = make_folds(patient_dirs, args.num_folds, args.seed)

    if is_main_process():
        with open(args.output_root / "shared_fold_split.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "split", "patient"])
            for i, (train_list, val_list) in enumerate(fold_defs):
                for p in train_list:
                    writer.writerow([i, "train", p.name])
                for p in val_list:
                    writer.writerow([i, "val", p.name])

    model_cfg_dict_template = load_model_cfg_dict_once(args.model_cfg)
    ks = parse_k_list(args.ks)
    root_rows = []

    for k in ks:
        prompt_map_k = load_prompt_map_for_k(args.prompt_xlsx, k)
        row = train_for_single_k(
            args,
            k,
            prompt_map_k,
            init_ckpt,
            model_cfg_dict_template,
            device,
            amp_dtype,
            use_ddp,
            local_rank,
            fold_defs,
        )
        if is_main_process() and row is not None:
            root_rows.append(row)
        if use_ddp:
            dist.barrier()

    if is_main_process():
        summary_csv = args.output_root / "k_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["k", "best_fold", "best_val_dice", "best_ckpt"])
            writer.writeheader()
            writer.writerows(root_rows)
        print("\n[DONE] K-loop training finished")
        print(f"[DONE] Summary: {summary_csv}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
