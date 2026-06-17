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
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

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


def window_to_uint8(img2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


class SAM2TrainPromptFromNii(SAM2Train):
    def __init__(self, *args, prompt_masks_by_video_id=None, **kwargs):
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
        self.prompt_masks_by_video_id = {
            int(k): torch.as_tensor(v, dtype=torch.bool) for k, v in (prompt_masks_by_video_id or {}).items()
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

        init_cond_frames = set()
        prompt_masks_per_object = []

        for obj_idx in range(o_dim):
            vid = int(obj_video_ids[obj_idx].item())
            prompt_vol = self.prompt_masks_by_video_id.get(vid, None)

            if prompt_vol is None:
                prompt_vol = torch.zeros((t_dim, masks_tohw.shape[2], masks_tohw.shape[3]), dtype=torch.bool)
            if prompt_vol.shape[0] != t_dim:
                if prompt_vol.shape[0] > t_dim:
                    prompt_vol = prompt_vol[:t_dim]
                else:
                    pad_t = t_dim - prompt_vol.shape[0]
                    prompt_vol = F.pad(prompt_vol, (0, 0, 0, 0, 0, pad_t))

            positive = torch.nonzero(prompt_vol.flatten(1).any(dim=1), as_tuple=False).flatten().tolist()
            if len(positive) == 0:
                positive = [int(start_frame_idx)]

            prompt_masks_per_object.append(prompt_vol)
            init_cond_frames.update(int(x) for x in positive)

        init_cond_frames = sorted(int(x) for x in init_cond_frames)
        init_set = set(init_cond_frames)
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [t for t in range(start_frame_idx, num_frames) if t not in init_set]

        backbone_out["mask_inputs_per_frame"] = {}
        for t in init_cond_frames:
            prompt_t = torch.zeros_like(gt_masks_per_frame[t])
            for o in range(o_dim):
                prompt_t[o, 0] = prompt_masks_per_object[o][t].to(prompt_t.device)
            backbone_out["mask_inputs_per_frame"][t] = prompt_t
        return backbone_out


class VolumeDataset(Dataset):
    def __init__(self, patient_dirs, image_name, mask_name, input_size, wc, ww):
        self.samples = []
        self.image_name = image_name
        self.mask_name = mask_name
        self.input_size = int(input_size)
        self.wc = float(wc)
        self.ww = float(ww)

        for pdir in patient_dirs:
            ip = pdir / image_name
            mp = pdir / mask_name
            if not ip.exists() or not mp.exists():
                continue
            pid = patient_id_from_folder(pdir)
            vid = patient_video_num_from_id(pid)
            self.samples.append((pdir, vid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pdir, vid = self.samples[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.image_name)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / self.mask_name)))
        gt = (gt > 0).astype(np.uint8)

        frames = []
        h0, w0 = img.shape[1], img.shape[2]
        for t in range(img.shape[0]):
            u8 = window_to_uint8(img[t], self.wc, self.ww)
            rgb = np.stack([u8, u8, u8], axis=0)
            image_tensor = torch.from_numpy(rgb).float() / 255.0
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            mask_tensor = torch.from_numpy(gt[t]).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, size=(self.input_size, self.input_size), mode="nearest")
            mask_tensor = mask_tensor.squeeze(0).squeeze(0).to(torch.bool)
            frames.append(Frame(data=image_tensor, objects=[Object(object_id=1, frame_index=t, segment=mask_tensor)]))

        return VideoDatapoint(frames=frames, video_id=int(vid), size=(h0, w0))


def load_prompt_volume_map_from_nii(patient_dirs, prompt_name: str, input_size: int):
    mapping = {}
    for pdir in patient_dirs:
        pid = patient_id_from_folder(pdir)
        vid = patient_video_num_from_id(pid)
        pp = pdir / prompt_name
        if not pp.exists():
            continue

        prompt_np = sitk.GetArrayFromImage(sitk.ReadImage(str(pp)))
        prompt_np = (prompt_np > 0).astype(np.float32)
        prompt_t = torch.from_numpy(prompt_np).unsqueeze(1)
        prompt_t = F.interpolate(prompt_t, size=(input_size, input_size), mode="nearest").squeeze(1).to(torch.bool)
        mapping[vid] = prompt_t
    return mapping


def load_model_cfg_dict_once(model_cfg: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_module("sam2", version_base="1.2"):
        cfg = compose(config_name=model_cfg)
    return OmegaConf.to_container(cfg.model, resolve=True)


def build_model(model_cfg_dict_template, init_ckpt: Path, prompt_masks_by_video_id, device):
    model_cfg_dict = dict(model_cfg_dict_template)

    image_encoder_cfg = model_cfg_dict.pop("image_encoder")
    memory_attention_cfg = model_cfg_dict.pop("memory_attention")
    memory_encoder_cfg = model_cfg_dict.pop("memory_encoder")
    model_cfg_dict.pop("_target_", None)

    image_encoder = instantiate(image_encoder_cfg, _recursive_=True)
    memory_attention = instantiate(memory_attention_cfg, _recursive_=True)
    memory_encoder = instantiate(memory_encoder_cfg, _recursive_=True)

    model = SAM2TrainPromptFromNii(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        prompt_masks_by_video_id=prompt_masks_by_video_id,
        **model_cfg_dict,
    )

    state = torch.load(str(init_ckpt), map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def freeze_image_encoder_only(model: nn.Module):
    trainable = 0
    frozen = 0
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if name.startswith("image_encoder"):
            p.requires_grad = False
            frozen += p.numel()
        else:
            p.requires_grad = True
            trainable += p.numel()

    if trainable == 0:
        raise RuntimeError("No trainable params left after freezing image_encoder.")

    print(f"[INFO] frozen image encoder params: {frozen}/{total} ({100.0 * frozen / max(total,1):.4f}%)")
    print(f"[INFO] trainable params: {trainable}/{total} ({100.0 * trainable / max(total,1):.4f}%)")


def build_optimizer(model, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


class DiceBCELoss(nn.Module):
    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        dice = (2.0 * inter + 1e-5) / (probs.sum() + targets.sum() + 1e-5)
        return 0.5 * bce + 0.5 * (1 - dice)


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


def run_epoch(model, loader, optimizer, scaler, device, amp_dtype, train_mode: bool):
    model.train(train_mode)
    criterion = DiceBCELoss().to(device)
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):
            outputs = model(batch)
            losses = []
            for t in range(batch.masks.shape[0]):
                logits = outputs[t]["pred_masks_high_res"][:, 0]
                losses.append(criterion(logits, batch.masks[t].float()))
            loss = torch.stack(losses).mean() if len(losses) else torch.zeros((), device=device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item())
        total_dice += compute_batch_volume_dice(outputs, batch.masks)
        n += 1

    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_dice / n


def make_folds(patient_dirs, num_folds: int, seed: int):
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
    parser = argparse.ArgumentParser("SAM2 5-fold training with prompt.nii.gz")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretrained-ckpt", type=Path, required=True)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--image-name", type=str, default="image.nii.gz")
    parser.add_argument("--mask-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--prompt-name", type=str, default="prompt.nii.gz")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--eta-min-factor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    set_seed(args.seed)

    if not args.train_root.exists():
        raise FileNotFoundError(f"train root not found: {args.train_root}")
    if not args.pretrained_ckpt.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrained_ckpt}")

    patient_dirs = sorted([p for p in args.train_root.iterdir() if p.is_dir()], key=patient_sort_key)
    if len(patient_dirs) < args.num_folds:
        raise ValueError(f"patients ({len(patient_dirs)}) < num_folds ({args.num_folds})")

    prompt_map = load_prompt_volume_map_from_nii(patient_dirs, args.prompt_name, args.input_size)
    if len(prompt_map) == 0:
        raise RuntimeError(f"No valid prompt masks found from {args.prompt_name}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    args.output_root.mkdir(parents=True, exist_ok=True)

    fold_defs = make_folds(patient_dirs, args.num_folds, args.seed)
    model_cfg_dict_template = load_model_cfg_dict_once(args.model_cfg)

    summary_rows = []
    for fold_idx, (train_patients, val_patients) in enumerate(fold_defs):
        print(f"\n[Fold {fold_idx}] train={len(train_patients)} val={len(val_patients)}")
        fold_dir = args.output_root / f"fold_{fold_idx}"
        ckpt_dir = fold_dir / "checkpoints"
        log_dir = fold_dir / "logs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        train_ds = VolumeDataset(train_patients, args.image_name, args.mask_name, args.input_size, args.window_center, args.window_width)
        val_ds = VolumeDataset(val_patients, args.image_name, args.mask_name, args.input_size, args.window_center, args.window_width)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=lambda b: collate_fn(b, dict_key="all"))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=lambda b: collate_fn(b, dict_key="all"))

        model = build_model(model_cfg_dict_template, args.pretrained_ckpt, prompt_map, device)
        freeze_image_encoder_only(model)
        optimizer = build_optimizer(model, args.lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.lr * args.eta_min_factor,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

        history = []
        best_val_dice = -1.0
        best_epoch = -1

        for epoch in range(args.epochs):
            tr_loss, tr_dice = run_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, True)
            with torch.no_grad():
                va_loss, va_dice = run_epoch(model, val_loader, optimizer, scaler, device, amp_dtype, False)

            lr_now = float(optimizer.param_groups[0]["lr"])
            history.append({
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "train_dice": tr_dice,
                "val_loss": va_loss,
                "val_dice": va_dice,
                "lr": lr_now,
            })
            print(f"[Fold {fold_idx}] epoch {epoch + 1}/{args.epochs} lr={lr_now:.6e} train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} val_loss={va_loss:.4f} val_dice={va_dice:.4f}")

            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_dice": va_dice}, str(ckpt_dir / "last.pth"))
            if va_dice > best_val_dice:
                best_val_dice = va_dice
                best_epoch = epoch + 1
                torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_dice": va_dice}, str(ckpt_dir / "best.pth"))

            scheduler.step()

        (log_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        summary_rows.append({
            "fold": fold_idx,
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "best_ckpt": str((ckpt_dir / "best.pth").resolve()),
            "last_ckpt": str((ckpt_dir / "last.pth").resolve()),
            "num_train_cases": len(train_ds),
            "num_val_cases": len(val_ds),
        })

    cv_csv = args.output_root / "cv_summary.csv"
    with open(cv_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "best_val_dice", "best_epoch", "best_ckpt", "last_ckpt", "num_train_cases", "num_val_cases"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    best_fold = max(summary_rows, key=lambda x: x["best_val_dice"])
    with open(args.output_root / "best_fold.txt", "w", encoding="utf-8") as f:
        f.write(f"best_fold: {best_fold['fold']}\n")
        f.write(f"best_val_dice: {best_fold['best_val_dice']:.6f}\n")
        f.write(f"best_epoch: {best_fold['best_epoch']}\n")
        f.write(f"best_ckpt: {best_fold['best_ckpt']}\n")

    print("\n[DONE] 5-fold training complete")
    print(f"[DONE] CV summary: {cv_csv}")
    print(f"[DONE] Best fold: {best_fold['fold']} | best_val_dice={best_fold['best_val_dice']:.4f}")


if __name__ == "__main__":
    main()
