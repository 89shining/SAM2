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
import pandas as pd
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


def normalize_patient_token(value) -> str:
    s = str(value).strip()
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    if m is None:
        return ""
    return f"CTV_{int(m.group(1)):03d}"


def patient_video_num_from_id(patient_id: str) -> int:
    m = re.search(r"(\d+)$", str(patient_id))
    if m is None:
        raise ValueError(f"Cannot parse numeric id from Patient_ID: {patient_id}")
    return int(m.group(1))


def parse_prompt_slices(text) -> list[int]:
    s = str(text).strip()
    if not s:
        return []
    return [int(x) for x in re.findall(r"-?\d+", s)]


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
    return mapping


def patient_id_from_folder(pdir: Path) -> str:
    m = re.search(r"(\d+)", pdir.name)
    if m is None:
        raise ValueError(f"Cannot parse patient id from folder name: {pdir.name}")
    return f"CTV_{int(m.group(1)):03d}"


def resolve_k_ckpt(train_output_root: Path, k: int) -> Path:
    k_root = train_output_root / f"K{k}"
    best_txt = k_root / "best_fold.txt"
    if best_txt.exists():
        m = re.search(r"best_ckpt:\s*(.+)", best_txt.read_text(encoding="utf-8", errors="ignore"))
        if m:
            p = Path(m.group(1).strip())
            if p.exists():
                return p

    cv_csv = k_root / "cv_summary.csv"
    if cv_csv.exists():
        best_row = None
        with open(cv_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    score = float(row["best_val_dice"])
                except Exception:
                    continue
                if best_row is None or score > best_row[0]:
                    best_row = (score, row["best_ckpt"])
        if best_row is not None:
            p = Path(best_row[1].strip())
            if p.exists():
                return p

    candidates = sorted(k_root.glob("fold_*/checkpoints/best.pth"))
    if len(candidates) > 0:
        return candidates[0]
    raise FileNotFoundError(f"No best checkpoint found under {k_root}")


@torch.no_grad()
def infer_with_prompt_list(predictor, frame_dir: Path, gt_zyx: np.ndarray, prompt_ids: list[int], obj_id: int):
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)

    z = gt_zyx.shape[0]
    valid_prompts = []
    for sid in prompt_ids:
        sid = int(max(0, min(int(sid), z - 1)))
        if (gt_zyx[sid] > 0).sum() > 0:
            valid_prompts.append(sid)
    valid_prompts = list(dict.fromkeys(valid_prompts))
    if len(valid_prompts) == 0:
        raise RuntimeError("No valid prompt slices after filtering")

    for sid in valid_prompts:
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=obj_id,
            mask=(gt_zyx[sid] > 0).astype(np.uint8),
        )

    _, h, w = gt_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)
    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == int(obj_id):
                pred[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break
    return pred, valid_prompts

def main():
    parser = argparse.ArgumentParser("Test one-shot multi-prompt K2..K10")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-output-root", type=Path, required=True)
    parser.add_argument("--prompt-xlsx", type=Path, required=True)
    parser.add_argument("--ks", type=str, default="2-10")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--img-name", type=str, default="image.nii.gz")
    parser.add_argument("--gt-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"test root not found: {args.test_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    ks = parse_k_list(args.ks)
    patient_dirs = sorted([p for p in args.test_root.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(patient_dirs)} test patients")

    all_k_summary = []
    for k in ks:
        ckpt_path = resolve_k_ckpt(args.train_output_root, k)
        prompt_map = load_prompt_map_for_k(args.prompt_xlsx, k)

        k_dir = args.output_root / f"K{k}"
        mask_dir = k_dir / "masks"
        k_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
        predictor = build_sam2_video_predictor(args.model_cfg, str(ckpt_path), device=device)

        rows = []
        for pdir in patient_dirs:
            patient_id = patient_id_from_folder(pdir)
            out_mask_path = mask_dir / f"{patient_id}.nii.gz"

            img_path = pdir / args.img_name
            gt_path = pdir / args.gt_name
            if not img_path.exists() or not gt_path.exists():
                print(f"[WARN] K{k} skip {pdir.name}: missing image or GT")
                continue

            img_zyx, img_sitk = read_nii_zyx(img_path)
            gt_zyx, _ = read_nii_zyx(gt_path)
            gt_zyx = (gt_zyx > 0).astype(np.uint8)
            if img_zyx.shape != gt_zyx.shape:
                print(f"[WARN] K{k} skip {pdir.name}: shape mismatch")
                continue

            vid = patient_video_num_from_id(patient_id)
            prompt_ids = prompt_map.get(vid, [])
            if len(prompt_ids) == 0:
                pos = np.where(gt_zyx.reshape(gt_zyx.shape[0], -1).any(axis=1))[0].tolist()
                if len(pos) == 0:
                    print(f"[WARN] K{k} skip {pdir.name}: empty GT")
                    continue
                lower, upper = int(min(pos)), int(max(pos))
                prompt_ids = [lower] if lower == upper else [lower, upper]

            tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_k{k}_{pdir.name}_"))
            try:
                save_frames_from_volume(img_zyx, tmp_dir, args.window_center, args.window_width)
                pred_zyx, used_prompts = infer_with_prompt_list(
                    predictor=predictor,
                    frame_dir=tmp_dir,
                    gt_zyx=gt_zyx,
                    prompt_ids=prompt_ids,
                    obj_id=args.obj_id,
                )
                write_mask_like(pred_zyx, img_sitk, out_mask_path)
                dsc = dice_3d(pred_zyx, gt_zyx)
                rows.append(
                    {
                        "Patient_ID": patient_id,
                        "K": k,
                        "PromptCount": len(used_prompts),
                        "PromptSlices": str(used_prompts),
                        "Dice3D": dsc,
                        "MaskPath": str(out_mask_path),
                    }
                )
                print(f"[OK] K{k} {pdir.name} dice={dsc:.4f} prompts={used_prompts}")
            except Exception as e:
                print(f"[WARN] K{k} {pdir.name} failed: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        detail_csv = k_dir / "metrics.csv"
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Patient_ID", "K", "PromptCount", "PromptSlices", "Dice3D", "MaskPath"],
            )
            writer.writeheader()
            writer.writerows(rows)

        mean_dice = float(np.mean([r["Dice3D"] for r in rows])) if len(rows) > 0 else 0.0
        all_k_summary.append(
            {
                "K": k,
                "Cases": len(rows),
                "MeanDice3D": mean_dice,
                "Checkpoint": str(ckpt_path),
                "DetailCSV": str(detail_csv),
            }
        )
        print(f"[K={k}] mean_dice={mean_dice:.4f} cases={len(rows)}")

    summary_csv = args.output_root / "k_test_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["K", "Cases", "MeanDice3D", "Checkpoint", "DetailCSV"])
        writer.writeheader()
        writer.writerows(all_k_summary)

    print(f"[DONE] Test summary: {summary_csv}")


if __name__ == "__main__":
    main()
