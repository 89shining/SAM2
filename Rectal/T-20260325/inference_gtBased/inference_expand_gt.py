#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 GT mask prompt 做 SAM2 推理：
按 GT 上下界进行 z 轴 crop，并同时测试多个外扩层数 margin。

每个 margin 会分别输出：
- two_mask_prompt/CTV_xxx.nii.gz
- three_mask_prompt/CTV_xxx.nii.gz
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
PROMPT_CSV = Path("/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/prompt_slices.csv")

OUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/gt_expand_experiments")

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

SAM2_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"

WINDOW_CENTER = 40
WINDOW_WIDTH = 400
OBJ_ID = 1

# 要同时跑的外扩层数
EXPAND_LIST = [0, 1, 3, 5, 7]


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


def to_int_or_none(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def normalize_prompt_slices(raw_slices, z_len):
    cleaned = []
    seen = set()
    for s in raw_slices:
        if s is None:
            continue
        if s < 0 or s >= z_len:
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def load_prompt_map(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt CSV 不存在: {csv_path}")

    prompt_map = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = (row.get("case_name") or "").strip()
            cid = extract_case_id(case_name)
            if cid is None:
                print(f"[WARN] 无法从 case_name 解析 ID，跳过: {case_name}")
                continue

            two_prompt = [
                to_int_or_none(row.get("two_prompt_z1")),
                to_int_or_none(row.get("two_prompt_z2")),
            ]
            three_prompt = [
                to_int_or_none(row.get("three_prompt_z1")),
                to_int_or_none(row.get("three_prompt_z2")),
                to_int_or_none(row.get("three_prompt_z3")),
            ]

            prompt_map[cid] = {
                "case_name": case_name,
                "two": two_prompt,
                "three": three_prompt,
            }
    return prompt_map


def get_boundary_from_prompt(prompt_slices):
    if len(prompt_slices) == 0:
        raise ValueError("prompt_slices 为空")
    return min(prompt_slices), max(prompt_slices)


def get_expand_crop_range(prompt_slices, z_full, margin):
    z_min, z_max = get_boundary_from_prompt(prompt_slices)
    z_start = max(0, z_min - margin)
    z_end = min(z_full - 1, z_max + margin)
    return z_start, z_end


def crop_volume_by_z(vol_zyx, z_start, z_end):
    return vol_zyx[z_start:z_end + 1]


def remap_prompt_slices_to_crop(prompt_slices, z_start, crop_len):
    new_slices = []
    seen = set()
    for s in prompt_slices:
        ns = s - z_start
        if 0 <= ns < crop_len and ns not in seen:
            new_slices.append(ns)
            seen.add(ns)
    return new_slices


# ========================= SAM2 推理 =========================
@torch.no_grad()
def sam2_infer_one_patient_crop(predictor, gt_crop_zyx, prompt_slices_crop, frame_dir):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    for s in prompt_slices_crop:
        mask = (gt_crop_zyx[s] > 0).astype(np.uint8)
        if mask.sum() == 0:
            print(f"[WARN] crop后提示层 {s} 的 GT mask 为空，跳过")
            continue
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(s),
            obj_id=OBJ_ID,
            mask=mask,
        )

    z, h, w = gt_crop_zyx.shape
    pred_crop = np.zeros((z, h, w), dtype=np.uint8)

    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == OBJ_ID:
                pred_crop[int(fidx)] = (logits[i] > 0).cpu().numpy().astype(np.uint8)
                break

    return pred_crop


def run_case_with_expand(predictor, img_zyx, gt_zyx, prompt_slices_full, margin):
    z_full = img_zyx.shape[0]
    z_start, z_end = get_expand_crop_range(prompt_slices_full, z_full, margin)

    img_crop = crop_volume_by_z(img_zyx, z_start, z_end)
    gt_crop = crop_volume_by_z(gt_zyx, z_start, z_end)

    crop_len = img_crop.shape[0]
    prompt_slices_crop = remap_prompt_slices_to_crop(prompt_slices_full, z_start, crop_len)

    if len(prompt_slices_crop) == 0:
        raise ValueError("crop 后提示层为空，无法推理")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_expand_{margin}_"))
    try:
        save_frames_from_volume(img_crop, tmp_dir)
        pred_crop = sam2_infer_one_patient_crop(
            predictor=predictor,
            gt_crop_zyx=gt_crop,
            prompt_slices_crop=prompt_slices_crop,
            frame_dir=tmp_dir,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    pred_full = np.zeros_like(gt_zyx, dtype=np.uint8)
    pred_full[z_start:z_end + 1] = pred_crop

    info = {
        "margin": margin,
        "z_start": z_start,
        "z_end": z_end,
        "prompt_full": prompt_slices_full,
        "prompt_crop": prompt_slices_crop,
    }
    return pred_full, info


# ========================= 主流程 =========================
def main():
    for margin in EXPAND_LIST:
        (OUT_ROOT / f"expand_{margin}" / "two_mask_prompt").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / f"expand_{margin}" / "three_mask_prompt").mkdir(parents=True, exist_ok=True)

    prompt_map = load_prompt_map(PROMPT_CSV)
    print(f"[INFO] Prompt rows loaded: {len(prompt_map)}")

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    predictor = build_sam2_video_predictor(SAM2_CFG, str(SAM2_CKPT), device=device)

    patient_dirs = [p for p in DATA_ROOT.iterdir() if p.is_dir()]
    print(f"[INFO] Found {len(patient_dirs)} patient folders")

    for pdir in patient_dirs:
        m = re.fullmatch(r"p_(\d+)", pdir.name)
        if m is None:
            print(f"[WARN] 非标准目录名，跳过: {pdir.name}")
            continue

        pid = int(m.group(1))
        if pid not in prompt_map:
            print(f"[WARN] CSV 中无 ID={pid}，跳过: {pdir.name}")
            continue

        img_path = pdir / IMG_NAME
        gt_path = pdir / GT_NAME
        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] 缺少 image/CTV，跳过: {pdir.name}")
            continue

        print(f"\n[INFO] Processing {pdir.name}")

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)

        if img_zyx.shape != gt_zyx.shape:
            print(f"[WARN] image 与 GT 形状不一致，跳过: {pdir.name}")
            continue

        row = prompt_map[pid]
        prompt_two = normalize_prompt_slices(row["two"], img_zyx.shape[0])
        prompt_three = normalize_prompt_slices(row["three"], img_zyx.shape[0])

        if len(prompt_two) < 2:
            print(f"[WARN] 两层提示不足2层，跳过: {pdir.name}, prompt_two={prompt_two}")
            continue
        if len(prompt_three) < 2:
            print(f"[WARN] 三层提示不足，跳过: {pdir.name}, prompt_three={prompt_three}")
            continue

        for margin in EXPAND_LIST:
            out_name = f"CTV_{pid:03d}.nii.gz"
            out_path_two = OUT_ROOT / f"expand_{margin}" / "two_mask_prompt" / out_name
            out_path_three = OUT_ROOT / f"expand_{margin}" / "three_mask_prompt" / out_name

            try:
                pred_two, info_two = run_case_with_expand(
                    predictor=predictor,
                    img_zyx=img_zyx,
                    gt_zyx=gt_zyx,
                    prompt_slices_full=prompt_two,
                    margin=margin,
                )
                write_mask_like(pred_two, img_sitk, out_path_two)

                pred_three, info_three = run_case_with_expand(
                    predictor=predictor,
                    img_zyx=img_zyx,
                    gt_zyx=gt_zyx,
                    prompt_slices_full=prompt_three,
                    margin=margin,
                )
                write_mask_like(pred_three, img_sitk, out_path_three)

                print(
                    f"[OK] margin={margin} | {pdir.name}\n"
                    f"     two   -> crop=({info_two['z_start']},{info_two['z_end']}) "
                    f"| full={info_two['prompt_full']} -> crop={info_two['prompt_crop']}\n"
                    f"     three -> crop=({info_three['z_start']},{info_three['z_end']}) "
                    f"| full={info_three['prompt_full']} -> crop={info_three['prompt_crop']}"
                )

            except Exception as e:
                print(f"[ERROR] margin={margin} | {pdir.name}: {e}")


if __name__ == "__main__":
    main()