#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2.1 5-fold cross-validation training (Medical CT)
- One patient = one video
- Prompt: ONLY upper & lower GT mask
- Freeze: image encoder
- Loss: Dice (logits)
- Save: train.log + training_curve.png
"""

import os
import sys
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# ------------------ SAM2 ------------------
sys.path.append("/home/wusi/sam2")   # ★ 确保是 sam2 repo 根目录
from sam2.build_sam import build_sam2


# =========================================================
# Config
# =========================================================

@dataclass
class TrainCfg:
    data_root: str

    image_name: str = "image.nii.gz"
    gt_name: str = "CTV.nii.gz"

    # ★ 必须是 sam2.1 对应的 config
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt: str = "/home/wusi/sam2/checkpoints/sam2.1_hiera_large.pt"

    device: str = "cuda"

    resize: int = 1024
    wl: float = 40
    ww: float = 400

    epochs: int = 50
    batch_size: int = 1
    num_workers: int = 4

    lr_decoder: float = 3e-4
    lr_other: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    max_frames: int = 0   # 0 = use full GT span

    patience: int = 6
    min_delta: float = 1e-4

    out_root: str = "./cv_out"
    seed: int = 2026
    n_folds: int = 5


# =========================================================
# Utils
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_level(img, wc, ww):
    img = img.astype(np.float32)
    low, high = wc - ww / 2, wc + ww / 2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-6)
    return (img * 255).astype(np.uint8)


def resize2d(img, size, is_mask=False):
    import cv2
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def find_gt_span(gt):
    z = np.where(gt.reshape(gt.shape[0], -1).sum(1) > 0)[0]
    return int(z[0]), int(z[-1])


# =========================================================
# Dataset
# =========================================================

class NiiVideoDataset(Dataset):
    def __init__(self, root, case_list: List[str], cfg: TrainCfg):
        self.root = Path(root)
        self.cases = case_list
        self.cfg = cfg

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.root / self.cases[idx]

        img = sitk.GetArrayFromImage(
            sitk.ReadImage(str(case / self.cfg.image_name))
        )
        gt = sitk.GetArrayFromImage(
            sitk.ReadImage(str(case / self.cfg.gt_name))
        )
        gt = (gt > 0).astype(np.uint8)

        z0, z1 = find_gt_span(gt)

        frames, masks = [], []
        for z in range(z0, z1 + 1):
            im = window_level(img[z], self.cfg.wl, self.cfg.ww)
            mk = gt[z]

            im = resize2d(im, self.cfg.resize)
            mk = resize2d(mk, self.cfg.resize, True)

            frames.append(np.stack([im] * 3))
            masks.append(mk[None])

        frames = torch.from_numpy(np.stack(frames)).float()
        masks = torch.from_numpy(np.stack(masks)).float()

        return {
            "images": frames,            # [T,3,H,W]
            "gt_masks": masks,           # [T,1,H,W]
            "upper_idx": 0,
            "lower_idx": frames.shape[0] - 1,
        }


def collate_fn(batch):
    return batch[0]


# =========================================================
# Training helpers
# =========================================================

def freeze_image_encoder(model):
    for name, p in model.named_parameters():
        if name.startswith("image_encoder."):
            p.requires_grad = False


def dice_loss_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((-2, -1))
    union = prob.sum((-2, -1)) + target.sum((-2, -1))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def sequence_loss(pred, gt):
    return sum(
        dice_loss_logits(pred[t:t+1], gt[t:t+1])
        for t in range(pred.shape[0])
    ) / pred.shape[0]


def forward_sam2(model, images, gt_masks, u, l):
    mask_prompts = {
        u: gt_masks[u],
        l: gt_masks[l],
    }
    out = model(images=images, mask_prompts=mask_prompts)
    for k in ["mask_logits", "pred_masks", "masks"]:
        if k in out:
            return out[k]
    raise RuntimeError("No mask logits found in SAM2 output")


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, val):
        if self.best is None or val < self.best - self.min_delta:
            self.best = val
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# =========================================================
# Train one fold
# =========================================================

def train_one_fold(cfg: TrainCfg, fold, train_ids, val_ids):

    out_dir = Path(cfg.out_root) / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- logger --------
    logger = logging.getLogger(f"fold_{fold}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh = logging.FileHandler(out_dir / "train.log")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"===== Fold {fold} start =====")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = build_sam2(
        config_file=cfg.model_cfg,
        ckpt_path=cfg.ckpt,
        device=device,
        mode="train",
        hydra_overrides_extra=[
            "++model._target_=training.model.sam2.SAM2Train"
        ],
    )

    freeze_image_encoder(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: total={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and "mask_decoder" in n],
             "lr": cfg.lr_decoder},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and "mask_decoder" not in n],
             "lr": cfg.lr_other},
        ],
        weight_decay=cfg.weight_decay,
    )

    train_ds = NiiVideoDataset(cfg.data_root, train_ids, cfg)
    val_ds = NiiVideoDataset(cfg.data_root, val_ids, cfg)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)

    stopper = EarlyStopping(cfg.patience, cfg.min_delta)

    epochs, tr_losses, va_losses = [], [], []
    best_val = 1e9

    for epoch in range(cfg.epochs):
        model.train()
        tl = []

        for b in train_loader:
            images = b["images"].to(device)
            gt = b["gt_masks"].to(device)

            optimizer.zero_grad()
            pred = forward_sam2(model, images, gt, b["upper_idx"], b["lower_idx"])
            loss = sequence_loss(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for b in val_loader:
                pred = forward_sam2(
                    model,
                    b["images"].to(device),
                    b["gt_masks"].to(device),
                    b["upper_idx"],
                    b["lower_idx"],
                )
                vl.append(sequence_loss(pred, b["gt_masks"].to(device)).item())

        tr_m, va_m = float(np.mean(tl)), float(np.mean(vl))
        epochs.append(epoch)
        tr_losses.append(tr_m)
        va_losses.append(va_m)

        logger.info(f"[Epoch {epoch:03d}] train={tr_m:.4f} val={va_m:.4f}")

        if va_m < best_val:
            best_val = va_m
            torch.save(model.state_dict(), out_dir / "best.pth")

        if stopper.step(va_m):
            logger.info("Early stopping triggered.")
            break

    # -------- plot --------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, tr_losses, label="Train")
    plt.plot(epochs, va_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=300)
    plt.close()

    logger.info(f"Fold {fold} finished. Best val={best_val:.4f}")


# =========================================================
# Main
# =========================================================

def main():
    cfg = TrainCfg(
        data_root="/home/wusi/SAMdata/Eso/20251217_CTV/cropnii_nnUNet/train_nii",
        out_root="/home/wusi/sam2/SAM2data/20260108/train-2slices",
    )

    set_seed(cfg.seed)

    patients = sorted([p.name for p in Path(cfg.data_root).iterdir() if p.is_dir()])

    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    for fold, (tr, va) in enumerate(kf.split(patients), start=1):
        train_ids = [patients[i] for i in tr]
        val_ids = [patients[i] for i in va]
        train_one_fold(cfg, fold, train_ids, val_ids)


if __name__ == "__main__":
    main()
