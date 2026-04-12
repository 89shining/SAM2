#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2 finetuning with 3 mask prompts (upper/lower + rule-middle).
Rule-middle follows Inference/rule_mask/mask_prompt_rule.py middle rule.

Example:
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_upper_lower_middle_rule_cv.py
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.distributed as dist
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

from sam2_train_upper_lower_middle_rule import SAM2TrainUpperLowerMiddleRuleMask
from training.loss_fns import MultiStepMultiMasksAndIous
from training.utils.data_utils import Frame, Object, VideoDatapoint, collate_fn


# ================= Default Paths (edit here) =================
DEFAULT_TRAIN_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii/train_nii")
DEFAULT_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/rule_mask/mask_prompt_3/two_epoch/TrainResult")
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


class UpperLowerMiddleRuleVolumeDataset(Dataset):
    """
    Build full-volume videos from 3D NIfTI:
    - each item is one patient volume (no clip split)
    - one object (CTV) per frame
    - prompts are computed inside model (upper/lower + rule-middle)
    """

    def __init__(
        self,
        patient_dirs,
        image_name="image.nii.gz",
        mask_name="CTV.nii.gz",
        window_center=40.0,
        window_width=400.0,
        input_size=1024,
    ):
        self.patient_dirs = list(patient_dirs)
        self.image_name = image_name
        self.mask_name = mask_name
        self.window_center = float(window_center)
        self.window_width = float(window_width)
        self.input_size = int(input_size)

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

            self.samples.append(pdir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pdir = self.samples[idx]
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
            video_id=idx,
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

    model = SAM2TrainUpperLowerMiddleRuleMask(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
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


def run_epoch_two_pass(model, loader, loss_fn, optimizer, scaler, device, amp_dtype, train_mode: bool):
    model.train(train_mode)
    total_loss = 0.0
    total_dice = 0.0
    n_batch = 0

    core_model = unwrap_model(model)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        # Pass-1: boundary prompts only
        core_model.set_middle_prompt_enabled(False)
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda"),
                dtype=amp_dtype,
            ):
                _ = model(batch)

        # Pass-2: boundary + rule-middle prompt
        core_model.set_middle_prompt_enabled(True)
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda"),
            dtype=amp_dtype,
        ):
            outputs_stage2 = model(batch)
            loss_dict = loss_fn(outputs_stage2, batch.masks)
            loss = loss_dict["core_loss"] if isinstance(loss_dict, dict) else loss_dict

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_dice += compute_batch_volume_dice(outputs_stage2, batch.masks)
        n_batch += 1

        core_model.set_middle_prompt_enabled(False)

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


def main():
    parser = argparse.ArgumentParser("5-fold SAM2 upper/lower->rule-middle iterative-mask prompt finetuning")
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT, help="Directory containing train patient folders")
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
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

        train_ds = UpperLowerMiddleRuleVolumeDataset(
            train_patients,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
        )
        val_ds = UpperLowerMiddleRuleVolumeDataset(
            val_patients,
            image_name=args.image_name,
            mask_name=args.mask_name,
            window_center=args.window_center,
            window_width=args.window_width,
            input_size=args.input_size,
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

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
