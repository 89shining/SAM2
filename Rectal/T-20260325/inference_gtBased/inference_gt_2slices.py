#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 GT 掩膜做 SAM2 推理：
在 GT 非空层范围 [z_min, z_max] 内（包含上下界）穷举任意两层作为提示，
对每组两层提示都跑一次推理，按与 GT 的 3D Dice 最高选择“最佳两层”。

输出：
- 最佳两层提示对应的分割：OUT_DIR / CTV_XXX.nii.gz
- 每例选择结果汇总：SUMMARY_CSV
"""

import csv
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

sys.path.append("/home/wusi/SAM2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam2.build_sam import build_sam2_video_predictor


# ===================== 路径与参数 =====================
DATA_ROOT = Path("/home/wusi/segment-anything/SAMdata/Rectal/20260310_CTV/datanii/test_nii")

OUT_DIR = Path("/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/two_mask_prompt_pick2_within_gt_range")
SUMMARY_CSV = OUT_DIR / "best_two_slices_summary.csv"

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

SAM2_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"  # "cuda" or "cpu"

WINDOW_CENTER = 40
WINDOW_WIDTH = 400

OBJ_ID = 1


# ========================= 工具函数 =========================
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


def read_nii_zyx(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_mask_like(pred_zyx, ref_img, out_path):
    pred_zyx = (pred_zyx > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(pred_zyx)
    out.SetSpacing(ref_img.GetSpacing())
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out, str(out_path))


def extract_case_id(text):
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    return int(nums[-1])


def find_first_last_nonzero(mask_3d):
    nz = np.where(mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1) > 0)[0]
    if len(nz) == 0:
        return None, None
    return int(nz[0]), int(nz[-1])


def dice3d(a, b, eps=1e-8):
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    sa = a.sum()
    sb = b.sum()
    if sa == 0 and sb == 0:
        return 1.0
    return float((2.0 * inter) / (sa + sb + eps))


# ========================= SAM2 推理 =========================
@torch.no_grad()
def sam2_infer_one_patient(predictor, gt_zyx, prompt_slices, frame_dir):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    for s in prompt_slices:
        mask = (gt_zyx[s] > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(s),
            obj_id=OBJ_ID,
            mask=mask,
        )

    z, h, w = gt_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)

    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == OBJ_ID:
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy().astype(np.uint8)
                break

    return pred


def choose_best_two_slices(predictor, gt_zyx, frame_dir):
    z_min, z_max = find_first_last_nonzero(gt_zyx)
    if z_min is None:
        return None, None, None, "GT empty"

    candidates = list(range(z_min, z_max + 1))
    if len(candidates) < 2:
        return None, None, None, f"GT non-empty slices < 2: [{z_min}, {z_max}]"

    best_pair = None
    best_pred = None
    best_dice = -1.0

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            pair = [candidates[i], candidates[j]]
            pred = sam2_infer_one_patient(
                predictor=predictor,
                gt_zyx=gt_zyx,
                prompt_slices=pair,
                frame_dir=frame_dir,
            )
            d = dice3d(pred, gt_zyx)

            # Dice 更高优先；若并列，优先跨度更大的 pair；再并列则更靠前的 pair
            if (
                d > best_dice
                or (
                    abs(d - best_dice) <= 1e-12
                    and best_pair is not None
                    and (pair[1] - pair[0] > best_pair[1] - best_pair[0])
                )
                or (
                    abs(d - best_dice) <= 1e-12
                    and best_pair is not None
                    and (pair[1] - pair[0] == best_pair[1] - best_pair[0])
                    and pair < best_pair
                )
            ):
                best_dice = d
                best_pair = pair
                best_pred = pred

    num_pairs = len(candidates) * (len(candidates) - 1) // 2
    return best_pair, best_pred, best_dice, f"searched {num_pairs} pairs in [{z_min}, {z_max}]"


# ========================= 主流程 =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)

    patient_dirs = [p for p in DATA_ROOT.iterdir() if p.is_dir()]
    print(f"[INFO] Found {len(patient_dirs)} patient folders")

    summary_rows = []

    for pdir in patient_dirs:
        m = re.fullmatch(r"p_(\d+)", pdir.name)
        if m is None:
            print(f"[WARN] 非标准目录名，跳过: {pdir.name}")
            continue

        pid = int(m.group(1))
        out_name = f"CTV_{pid:03d}.nii.gz"
        out_path = OUT_DIR / out_name

        img_path = pdir / IMG_NAME
        gt_path = pdir / GT_NAME
        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] 缺少 image/CTV，跳过: {pdir.name}")
            continue

        print(f"[INFO] Processing {pdir.name} -> {out_name}")

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)
        gt_zyx = (gt_zyx > 0).astype(np.uint8)

        if img_zyx.shape != gt_zyx.shape:
            print(f"[WARN] image 与 GT 形状不一致，跳过: {pdir.name}")
            continue

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_pick2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            best_pair, best_pred, best_dice, note = choose_best_two_slices(
                predictor=predictor,
                gt_zyx=gt_zyx,
                frame_dir=tmp_dir,
            )

            if best_pair is None:
                print(f"[WARN] {pdir.name} 无法选择两层提示: {note}")
                summary_rows.append(
                    {
                        "case_name": pdir.name,
                        "case_id": pid,
                        "best_prompt_z1": "",
                        "best_prompt_z2": "",
                        "best_dice": "",
                        "note": note,
                    }
                )
                continue

            write_mask_like(best_pred, img_sitk, out_path)
            print(
                f"[OK] {pdir.name} saved: {out_path.name} "
                f"| best_pair={best_pair} | dice={best_dice:.6f} | {note}"
            )

            summary_rows.append(
                {
                    "case_name": pdir.name,
                    "case_id": pid,
                    "best_prompt_z1": best_pair[0],
                    "best_prompt_z2": best_pair[1],
                    "best_dice": f"{best_dice:.6f}",
                    "note": note,
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "case_id",
                "best_prompt_z1",
                "best_prompt_z2",
                "best_dice",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] Summary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
