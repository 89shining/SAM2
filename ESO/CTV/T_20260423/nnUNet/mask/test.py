#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "training").is_dir() and (p / "sam2").is_dir():
            return p
    raise RuntimeError("Cannot locate SAM2 project root")


PROJECT_ROOT = _find_project_root(CURRENT_DIR)
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2.build_sam import build_sam2_video_predictor

DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def window_to_uint8(img2d: np.ndarray, wc: float, ww: float) -> np.ndarray:
    img = img2d.astype(np.float32)
    lo = wc - ww / 2.0
    hi = wc + ww / 2.0
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6) * 255.0
    return img.astype(np.uint8)


def save_frames_from_volume(vol_zyx: np.ndarray, out_dir: Path, wc: float, ww: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(vol_zyx.shape[0]):
        u8 = window_to_uint8(vol_zyx[i], wc, ww)
        rgb = np.stack([u8, u8, u8], axis=-1)
        Image.fromarray(rgb).save(out_dir / f"{i:05d}.jpg", quality=95)


def read_nii_zyx(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def write_mask_like(pred_zyx: np.ndarray, ref_img: sitk.Image, out_path: Path):
    out = sitk.GetImageFromArray((pred_zyx > 0).astype(np.uint8))
    out.SetSpacing(ref_img.GetSpacing())
    out.SetOrigin(ref_img.GetOrigin())
    out.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out, str(out_path))


def dice_3d(pred: np.ndarray, gt: np.ndarray, eps=1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


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


def patient_sort_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def resolve_best_ckpt(train_output_root: Path) -> tuple[int, Path]:
    best_txt = train_output_root / "best_fold.txt"
    if best_txt.exists():
        txt = best_txt.read_text(encoding="utf-8", errors="ignore")
        m_fold = re.search(r"best_fold:\s*(\d+)", txt)
        m_ckpt = re.search(r"best_ckpt:\s*(.+)", txt)
        if m_fold and m_ckpt:
            ckpt = Path(m_ckpt.group(1).strip())
            if ckpt.exists():
                return int(m_fold.group(1)), ckpt

    cv_csv = train_output_root / "cv_summary.csv"
    if cv_csv.exists():
        best = None
        with open(cv_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    score = float(row["best_val_dice"])
                    fold = int(row["fold"])
                    ckpt = Path(row["best_ckpt"].strip())
                except Exception:
                    continue
                if not ckpt.exists():
                    continue
                if best is None or score > best[0]:
                    best = (score, fold, ckpt)
        if best is not None:
            return best[1], best[2]

    candidates = sorted(train_output_root.glob("fold_*/checkpoints/best.pth"))
    if len(candidates) > 0:
        m = re.search(r"fold_(\d+)", str(candidates[0]))
        fold = int(m.group(1)) if m else -1
        return fold, candidates[0]

    raise FileNotFoundError(f"No best checkpoint found under {train_output_root}")


def build_prompt_mask_dict(prompt_zyx: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for z in range(prompt_zyx.shape[0]):
        m = (prompt_zyx[z] > 0).astype(np.uint8)
        if m.sum() > 0:
            out[z] = m
    return out


@torch.no_grad()
def infer_with_prompt_mask_nii(predictor, frame_dir: Path, prompt_zyx: np.ndarray, obj_id: int):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    prompt_dict = build_prompt_mask_dict(prompt_zyx)
    if len(prompt_dict) == 0:
        raise RuntimeError("No valid prompt slices from prompt.nii.gz")

    for sid in sorted(prompt_dict.keys()):
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(sid),
            obj_id=int(obj_id),
            mask=prompt_dict[sid],
        )

    z, h, w = prompt_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)
    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == int(obj_id):
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy().astype(np.uint8)
                break
    return pred, sorted(prompt_dict.keys())


def main():
    parser = argparse.ArgumentParser("Test with best fold checkpoint (prompt.nii.gz)")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--train-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--img-name", type=str, default="image.nii.gz")
    parser.add_argument("--gt-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--prompt-name", type=str, default="prompt.nii.gz")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"test root not found: {args.test_root}")

    best_fold, ckpt_path = resolve_best_ckpt(args.train_output_root)
    print(f"[INFO] best fold: {best_fold}")
    print(f"[INFO] ckpt: {ckpt_path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    predictor = build_sam2_video_predictor(args.model_cfg, str(ckpt_path), device=device)

    patient_dirs = sorted([p for p in args.test_root.iterdir() if p.is_dir()], key=patient_sort_key)
    rows = []

    for pdir in patient_dirs:
        pid = patient_id_from_folder(pdir)
        vid = patient_video_num_from_id(pid)
        img_path = pdir / args.img_name
        gt_path = pdir / args.gt_name
        prompt_path = pdir / args.prompt_name

        if (not img_path.exists()) or (not prompt_path.exists()):
            print(f"[WARN] skip {pdir.name} (missing image/prompt)")
            continue

        img_zyx, ref_img = read_nii_zyx(img_path)
        prompt_zyx, _ = read_nii_zyx(prompt_path)

        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as td:
            frame_dir = Path(td)
            save_frames_from_volume(img_zyx, frame_dir, args.window_center, args.window_width)
            pred_zyx, used_prompts = infer_with_prompt_mask_nii(
                predictor,
                frame_dir,
                prompt_zyx,
                obj_id=args.obj_id,
            )

        out_path = mask_dir / f"{pid}.nii.gz"
        write_mask_like(pred_zyx, ref_img, out_path)

        row = {
            "patient": pdir.name,
            "patient_id": pid,
            "video_id": vid,
            "pred_path": str(out_path.resolve()),
            "prompt_slices": " ".join(str(x) for x in used_prompts),
        }

        if gt_path.exists():
            gt_zyx, _ = read_nii_zyx(gt_path)
            score = dice_3d(pred_zyx > 0, gt_zyx > 0)
            row["dice"] = score
            print(f"[TEST] {pdir.name} dice={score:.4f}")
        else:
            row["dice"] = ""
            print(f"[TEST] {pdir.name} done (no GT for dice)")

        rows.append(row)

    csv_path = args.output_root / "test_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient", "patient_id", "video_id", "dice", "prompt_slices", "pred_path"])
        writer.writeheader()
        writer.writerows(rows)

    valid_dice = [float(r["dice"]) for r in rows if r["dice"] != ""]
    if len(valid_dice) > 0:
        mean_dice = float(np.mean(valid_dice))
        print(f"[DONE] mean dice={mean_dice:.4f}")

    print(f"[DONE] saved masks: {mask_dir}")
    print(f"[DONE] csv: {csv_path}")


if __name__ == "__main__":
    main()
