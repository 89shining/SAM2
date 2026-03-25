#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 GT 的 CTV 掩膜作为提示层进行 SAM2 推理。

输入：完整的CT体积
- DATA_ROOT 下每个病例目录（命名示例：p_10），每例包含 image.nii.gz 与 CTV.nii.gz。
- PROMPT_CSV（由 slice_select.py 生成），包含每例的两层/三层提示 z 索引。

输出：
- 两层提示结果：OUT_DIR_TWO / CTV_XXX.nii.gz
- 三层提示结果：OUT_DIR_THREE / CTV_XXX.nii.gz

关键保证：
- 按“病例数字 ID”对齐 CSV 与病例目录，不依赖排序位置。
- 例如 p_10 一定对应 CSV 里 ID=10 的那一行，不会出现 10 对应到 2。
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

# slice_select.py 生成的 CSV
PROMPT_CSV = Path("/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/prompt_slices.csv")

OUT_DIR_TWO = Path("/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/two_mask_prompt")
OUT_DIR_THREE = Path("/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/three_mask_prompt")

IMG_NAME = "image.nii.gz"
GT_NAME = "CTV.nii.gz"

SAM2_CKPT = Path("/home/wusi/SAM2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DEVICE = "cuda"  # "cuda" or "cpu"

# CT window
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
    """从字符串中提取病例数字 ID（取最后一段数字）。"""
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
    """清理提示层：转 int、去重、过滤越界。"""
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
    """
    读取 CSV，返回：{case_id: {two: [...], three: [...]}}
    case_id 从 case_name 中解析得到。
    """
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

            if cid in prompt_map:
                print(f"[WARN] CSV 中 ID={cid} 重复，使用后出现的一行。")

            prompt_map[cid] = {
                "case_name": case_name,
                "two": two_prompt,
                "three": three_prompt,
            }

    return prompt_map


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
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break

    return pred


# ========================= 主流程 =========================
def main():
    OUT_DIR_TWO.mkdir(parents=True, exist_ok=True)
    OUT_DIR_THREE.mkdir(parents=True, exist_ok=True)

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

        out_name = f"CTV_{pid:03d}.nii.gz"
        out_path_two = OUT_DIR_TWO / out_name
        out_path_three = OUT_DIR_THREE / out_name

        img_path = pdir / IMG_NAME
        gt_path = pdir / GT_NAME
        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] 缺少 image/CTV，跳过: {pdir.name}")
            continue

        print(f"[INFO] Processing {pdir.name} -> {out_name}")

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)

        if img_zyx.shape != gt_zyx.shape:
            print(f"[WARN] image 与 GT 形状不一致，跳过: {pdir.name}")
            continue

        row = prompt_map[pid]
        prompt_two = normalize_prompt_slices(row["two"], img_zyx.shape[0])
        prompt_three = normalize_prompt_slices(row["three"], img_zyx.shape[0])

        if len(prompt_two) == 0:
            print(f"[WARN] 两层提示为空，跳过该病例: {pdir.name}")
            continue
        if len(prompt_three) == 0:
            print(f"[WARN] 三层提示为空，跳过该病例: {pdir.name}")
            continue

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir)

            # 两层提示推理
            pred_two = sam2_infer_one_patient(
                predictor=predictor,
                gt_zyx=gt_zyx,
                prompt_slices=prompt_two,
                frame_dir=tmp_dir,
            )
            write_mask_like(pred_two, img_sitk, out_path_two)

            # 三层提示推理
            pred_three = sam2_infer_one_patient(
                predictor=predictor,
                gt_zyx=gt_zyx,
                prompt_slices=prompt_three,
                frame_dir=tmp_dir,
            )
            write_mask_like(pred_three, img_sitk, out_path_three)

            print(
                f"[OK] {pdir.name} saved: two={out_path_two.name}({prompt_two}) "
                f"three={out_path_three.name}({prompt_three})"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
