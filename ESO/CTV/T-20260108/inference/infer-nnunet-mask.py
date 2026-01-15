"""
用nnUNet的预测结果做mask提示，其他设置与inference-mask中相同
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2 inference on 3D NIfTI volumes (Z as video frames),
nnUNet mask as prompt (imperfect prior),
batch over patients.

Prediction naming rule:
  p_10 -> CTV_010.nii.gz

All predictions are saved to ONE output directory.
"""

import os
import sys
sys.path.append("/home/wusi/sam2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch

from sam2.build_sam import build_sam2_video_predictor


# ======================================================
# ================ 路径 & 配置（只改这里） ==============
# ======================================================

DATA_ROOT = Path(
    "/home/wusi/SAMdata/Eso/20260104_CTV/nnUNet_mask/cropdatanii/test_nii"
)

OUT_DIR = Path(
    "/home/wusi/sam2/SAM2data/20260108/inference_mask/nnUNet-prompt/2s_mask_nnunet"
)

IMG_NAME     = "image.nii.gz"
GT_NAME      = "CTV.nii.gz"              # 仅用于评估
NNUNET_NAME  = "prompt.nii.gz"      # ★ 作为 SAM2 的 mask prompt

SAM2_CKPT = Path(
    "/home/wusi/sam2/checkpoints/sam2.1_hiera_large.pt"
)
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"   # "cuda" or "cpu"

# CT window
WINDOW_CENTER = 40
WINDOW_WIDTH  = 400

# prompt setting（与 GT prompt 实验保持完全一致）
PROMPT_MODE   = "uniform"   # uniform / endpoints / custom
NUM_PROMPTS   = 2
CUSTOM_SLICES = ""          # e.g. "10,25,40"

OBJ_ID = 1


# ======================================================
# ======================= Utils =========================
# ======================================================

def window_to_uint8(img2d, wc, ww):
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def save_frames_from_volume(vol_zyx, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(vol_zyx.shape[0]):
        u8 = window_to_uint8(vol_zyx[i], WINDOW_CENTER, WINDOW_WIDTH)
        rgb = np.stack([u8, u8, u8], axis=-1)
        Image.fromarray(rgb).save(out_dir / f"{i:05d}.jpg", quality=95)


def choose_prompt_slices(z_len):
    if PROMPT_MODE == "custom":
        return [int(s) for s in CUSTOM_SLICES.split(",") if s.strip().isdigit()]

    if NUM_PROMPTS == 1:
        return [z_len // 2]

    idx = np.linspace(0, z_len - 1, NUM_PROMPTS).round().astype(int).tolist()
    return sorted(set(idx))


def read_nii_zyx(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return arr, img


def write_mask_like(pred_zyx, ref_img, out_path):
    pred_zyx = (pred_zyx > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(pred_zyx)
    out.SetSpacing(ref_img.GetSpacing())
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out, str(out_path))


# ======================================================
# ================== SAM2 推理 ==========================
# ======================================================

@torch.no_grad()
def sam2_infer_one_patient(
    predictor,
    img_zyx,
    prompt_mask_zyx,   # ★ nnUNet mask（imperfect prior）
    prompt_slices,
    frame_dir,
):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    for s in prompt_slices:
        mask = (prompt_mask_zyx[s] > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(s),
            obj_id=OBJ_ID,
            mask=mask,
        )

    z, h, w = img_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)

    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == OBJ_ID:
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break

    return pred


# ======================================================
# ======================== Main =========================
# ======================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    predictor = build_sam2_video_predictor(
        SAM2_CFG, str(SAM2_CKPT), device=device
    )

    patient_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"[INFO] Found {len(patient_dirs)} patients")

    for pdir in patient_dirs:
        m = re.fullmatch(r"p_(\d+)", pdir.name)
        if m is None:
            raise RuntimeError(f"Invalid folder name: {pdir.name}")

        pid = int(m.group(1))
        out_name = f"CTV_{pid:03d}.nii.gz"
        out_path = OUT_DIR / out_name

        img_path     = pdir / IMG_NAME
        gt_path      = pdir / GT_NAME
        nnunet_path  = pdir / NNUNET_NAME

        if not img_path.exists() or not gt_path.exists() or not nnunet_path.exists():
            print(f"[WARN] Skip {pdir.name}")
            continue

        print(f"[INFO] {pdir.name} -> {out_name}")

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _         = read_nii_zyx(gt_path)
        nnunet_zyx, _     = read_nii_zyx(nnunet_path)

        # 强制一致性检查（非常重要）
        assert img_zyx.shape == nnunet_zyx.shape, \
            f"Shape mismatch: image {img_zyx.shape}, nnUNet {nnunet_zyx.shape}"

        prompt_slices = choose_prompt_slices(img_zyx.shape[0])

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            pred = sam2_infer_one_patient(
                predictor,
                img_zyx,
                nnunet_zyx,   # ★ 唯一变化：nnUNet mask 作为 prompt
                prompt_slices,
                tmp_dir,
            )

            write_mask_like(pred, img_sitk, out_path)
            print(f"[OK] Saved {out_path}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
