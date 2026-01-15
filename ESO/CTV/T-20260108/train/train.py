#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM2 2D slice-level finetuning (SAM1-style)
- No video / no memory pipeline
- Prompt: nnUNet mask as mask prompt (prompt.nii.gz)
- GT: CTV.nii.gz
- Supervise on low-res logits (256x256)
- 5-fold CV, logs + curves
- Early stopping (NEW)
"""

import os
import sys

sys.path.append("/home/wusi/sam2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import random
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from sam2.build_sam import build_sam2


# =========================
# Config
# =========================
ROOT_DIR = Path("/home/wusi/SAMdata/Eso/20260104_CTV/nnUNet_mask/cropdatanii/train_nii")

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT   = "/home/wusi/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

SAVE_ROOT = Path("/home/wusi/sam2/SAM2data/20260108/TrainResult")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 1024
LOW_RES  = 256

EPOCHS = 100
BATCH_SIZE = 12
LR = 1e-4
NUM_WORKERS = 4
N_FOLDS = 5
SEED = 42

WL_CENTER = 40
WL_WIDTH  = 400

# ===== Early stopping =====
PATIENCE = 10
MIN_DELTA = 1e-3


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_level(img2d: np.ndarray, center=40, width=400) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo = center - width / 2
    hi = center + width / 2
    img = np.clip(img, lo, hi)
    img = (img - lo) / width * 255.0
    return img.astype(np.uint8)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, smooth=1e-5) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = bce.mean(dim=(1, 2, 3))

    return (dice + bce).mean()


@torch.no_grad()
def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth=1e-5) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return float(dice.mean().item())


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =========================
# Dataset
# =========================
class SliceDataset(Dataset):
    def __init__(self, root_dir: Path, patient_ids: list[str]):
        self.samples = []

        for pid in patient_ids:
            pdir = root_dir / pid
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / "image.nii.gz")))
            gt  = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / "CTV.nii.gz")))
            pm  = sitk.GetArrayFromImage(sitk.ReadImage(str(pdir / "prompt.nii.gz")))

            for z in range(img.shape[0]):
                if (gt[z] > 0).sum() == 0:
                    continue
                self.samples.append((img[z], gt[z], pm[z]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img2d, gt2d, pm2d = self.samples[idx]

        img2d = window_level(img2d, WL_CENTER, WL_WIDTH)
        img = Image.fromarray(img2d).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).float().unsqueeze(0).repeat(3, 1, 1)

        gt = Image.fromarray((gt2d > 0).astype(np.uint8) * 255).resize(
            (LOW_RES, LOW_RES), Image.NEAREST)
        gt = torch.from_numpy((np.array(gt) > 0).astype(np.float32)).unsqueeze(0)

        pm = Image.fromarray((pm2d > 0).astype(np.uint8) * 255).resize(
            (LOW_RES, LOW_RES), Image.NEAREST)
        pm = torch.from_numpy((np.array(pm) > 0).astype(np.float32)).unsqueeze(0)

        return {"image": img, "gt": gt, "mask_prompt": pm}


# =========================
# SAM2 forward (unchanged)
# =========================
def sam2_forward_2d(model, image, mask_prompt):
    backbone_out = model.forward_image(image)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    B = image.size(0)
    C = model.hidden_dim
    H, W = feat_sizes[-1]
    pix_feat = vision_feats[-1].permute(1, 2, 0).contiguous().view(B, C, H, W)

    sparse_emb, dense_emb = model.sam_prompt_encoder(
        points=None,
        boxes=None,
        masks=mask_prompt
    )

    if getattr(model, "use_high_res_features_in_sam", False):
        high_res_features = [
            backbone_out["backbone_fpn"][0],
            backbone_out["backbone_fpn"][1]
        ]
    else:
        high_res_features = None

    low_res_multimasks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=pix_feat,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features
    )

    return low_res_multimasks


# =========================
# Train one fold
# =========================
def train_one_fold(fold_id: int, train_pids: list[str], val_pids: list[str]):
    fold_dir = SAVE_ROOT / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(fold_dir / "train.log")

    train_ds = SliceDataset(ROOT_DIR, train_pids)
    val_ds   = SliceDataset(ROOT_DIR, val_pids)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE).to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=1e-4
    )

    best_val_dice = -1.0
    patience_counter = 0

    train_losses, val_dices = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch:03d}"):
            img = batch["image"].to(DEVICE)
            gt  = batch["gt"].to(DEVICE)
            pm  = batch["mask_prompt"].to(DEVICE)

            optimizer.zero_grad()
            logits = sam2_forward_2d(model, img, pm)
            loss = dice_bce_loss(logits, gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                gt  = batch["gt"].to(DEVICE)
                pm  = batch["mask_prompt"].to(DEVICE)
                logits = sam2_forward_2d(model, img, pm)
                dices.append(dice_from_logits(logits, gt))

        val_dice = float(np.mean(dices))
        val_dices.append(val_dice)

        logger.info(f"Epoch {epoch:03d} | train_loss={epoch_loss:.6f} | val_dice={val_dice:.6f}")

        if val_dice > best_val_dice + MIN_DELTA:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), fold_dir / "best.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch:03d}")
                break

    return best_val_dice


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    patients = sorted([d.name for d in ROOT_DIR.iterdir() if d.is_dir()])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_i, (tr, va) in enumerate(kf.split(patients), 1):
        train_pids = [patients[i] for i in tr]
        val_pids   = [patients[i] for i in va]
        train_one_fold(fold_i, train_pids, val_pids)

    print("All folds finished.")


if __name__ == "__main__":
    main()
