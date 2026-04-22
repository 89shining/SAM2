#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from medpy.metric.binary import hd95 as medpy_hd95
from openpyxl import Workbook
from PIL import Image

# keep local import robust
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

from sam2.build_sam import build_sam2_video_predictor


DEFAULT_TEST_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/datanii/test_nii")
DEFAULT_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/BadHD95_slice/mask_prompt_3/two_epoch/TestResult")
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_FINETUNED_CKPT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/BadHD95_slice/mask_prompt_3/two_epoch/TrainResult/fold_0/checkpoints/best.pth")
DEFAULT_TRAIN_OUTPUT_ROOT = Path("/home/wusi/SAM2/SAM2data/Eso/20260326/Train/BadHD95_slice/mask_prompt_3/two_epoch/TrainResult")


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
    pred_zyx = (pred_zyx > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(pred_zyx)
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


def gt_positive_slices(gt_zyx: np.ndarray):
    non_empty = np.where(gt_zyx.reshape(gt_zyx.shape[0], -1).any(axis=1))[0]
    return [int(z) for z in non_empty.tolist()]


def safe_hd95_2d(pred2d: np.ndarray, gt2d: np.ndarray) -> float:
    pred2d = pred2d.astype(bool)
    gt2d = gt2d.astype(bool)
    if pred2d.sum() == 0 or gt2d.sum() == 0:
        return -1.0
    try:
        return float(medpy_hd95(pred2d, gt2d, voxelspacing=(1.0, 1.0)))
    except Exception:
        return -1.0


def select_middle_from_pred_and_gt_hd95(pred_zyx: np.ndarray, gt_zyx: np.ndarray, lower_id: int, upper_id: int) -> int:
    candidate_slices = list(range(lower_id + 1, upper_id))
    if len(candidate_slices) == 0:
        return int(lower_id)
    valid_scores = []
    for z in candidate_slices:
        score = safe_hd95_2d(pred_zyx[z], gt_zyx[z])
        if score >= 0:
            valid_scores.append((z, score))
    if len(valid_scores) == 0:
        return int(lower_id)
    return int(max(valid_scores, key=lambda x: x[1])[0])


def _propagate_to_mask(state, predictor, gt_zyx: np.ndarray, obj_id: int) -> np.ndarray:
    z, h, w = gt_zyx.shape
    pred = np.zeros((z, h, w), dtype=np.uint8)
    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == obj_id:
                pred_i = logits[i]
                if pred_i.ndim == 3 and pred_i.shape[0] == 1:
                    pred_i = pred_i[0]
                pred[int(fidx)] = (pred_i > 0).detach().cpu().numpy().astype(np.uint8)
                break
    return pred


@torch.no_grad()
def infer_two_stage_iterative(
    predictor,
    frame_dir: Path,
    gt_zyx: np.ndarray,
    lower_id: int,
    upper_id: int,
    obj_id: int,
):
    # Stage-1: boundary prompts
    state = predictor.init_state(video_path=str(frame_dir))
    predictor.reset_state(state)
    stage1_prompt_ids = sorted(set([int(lower_id), int(upper_id)]))
    for sid in stage1_prompt_ids:
        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=obj_id,
            mask=prompt_mask,
        )
    pred_stage1 = _propagate_to_mask(state, predictor, gt_zyx, obj_id)

    # Online bad_95 middle selection from stage-1 output
    middle_id = select_middle_from_pred_and_gt_hd95(
        pred_zyx=pred_stage1,
        gt_zyx=gt_zyx,
        lower_id=lower_id,
        upper_id=upper_id,
    )

    # Stage-2: iterative memory inheritance + middle prompt injection
    if int(middle_id) not in stage1_prompt_ids:
        middle_mask = (gt_zyx[int(middle_id)] > 0).astype(np.uint8)
        if middle_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {middle_id} is empty in GT.")
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(middle_id),
            obj_id=obj_id,
            mask=middle_mask,
        )
    pred_stage2 = _propagate_to_mask(state, predictor, gt_zyx, obj_id)
    return pred_stage1, pred_stage2, middle_id


def patient_id_from_folder(pdir: Path):
    m = re.search(r"\d+", pdir.name)
    if m is None:
        raise ValueError(f"Cannot parse patient id from folder name: {pdir.name}")
    return f"CTV_{int(m.group()):03d}"


def resolve_ckpt(finetuned_ckpt: Path, train_output_root: Path) -> Path:
    # Use training best fold checkpoint first.
    best_fold_txt = train_output_root / "best_fold.txt"
    if best_fold_txt.exists():
        content = best_fold_txt.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"best_ckpt:\s*(.+)", content)
        if m:
            best_ckpt = Path(m.group(1).strip())
            if best_ckpt.exists():
                return best_ckpt

    if finetuned_ckpt.exists():
        return finetuned_ckpt

    if DEFAULT_FINETUNED_CKPT.exists():
        return DEFAULT_FINETUNED_CKPT

    raise FileNotFoundError(
        "No usable finetuned checkpoint found. "
        f"Tried: best_fold.txt in {train_output_root}, {finetuned_ckpt}, {DEFAULT_FINETUNED_CKPT}"
    )


def main():
    parser = argparse.ArgumentParser("Test SAM2 with online bad_95 iterative two-stage prompts")
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT, help="Separate test set root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Save root for masks/excel")
    parser.add_argument("--finetuned-ckpt", type=Path, default=DEFAULT_FINETUNED_CKPT, help="Fallback checkpoint for inference")
    parser.add_argument("--train-output-root", type=Path, default=DEFAULT_TRAIN_OUTPUT_ROOT, help="Training output root for auto resolving best fold ckpt")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--img-name", type=str, default="image.nii.gz")
    parser.add_argument("--gt-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--excel-name", type=str, default="two_epoch_iterative_bad95_results.xlsx")
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"test root not found: {args.test_root}")
    ckpt_path = resolve_ckpt(args.finetuned_ckpt, args.train_output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    best_mask_dir = args.output_root / "best_mask"
    best_mask_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = args.output_root / args.excel_name

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(ckpt_path),
        device=device,
    )
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    patient_dirs = sorted([p for p in args.test_root.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(patient_dirs)} patients")

    rows = []
    for pdir in patient_dirs:
        patient_id = patient_id_from_folder(pdir)
        out_mask_path = best_mask_dir / f"{patient_id}.nii.gz"
        img_path = pdir / args.img_name
        gt_path = pdir / args.gt_name

        if not img_path.exists() or not gt_path.exists():
            print(f"[WARN] Skip {pdir.name}: missing image or GT")
            continue

        img_zyx, img_sitk = read_nii_zyx(img_path)
        gt_zyx, _ = read_nii_zyx(gt_path)
        gt_zyx = (gt_zyx > 0).astype(np.uint8)

        if img_zyx.shape != gt_zyx.shape:
            print(f"[WARN] Skip {pdir.name}: shape mismatch img{img_zyx.shape} vs gt{gt_zyx.shape}")
            continue

        pos = gt_positive_slices(gt_zyx)
        if len(pos) == 0:
            print(f"[WARN] Skip {pdir.name}: GT has no positive slices")
            continue
        lower_id = int(min(pos))
        upper_id = int(max(pos))

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_test_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir, args.window_center, args.window_width)
            pred_stage1, pred_stage2, middle_id = infer_two_stage_iterative(
                predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                obj_id=args.obj_id,
            )
            dice_stage1 = dice_3d(pred_stage1, gt_zyx)
            dice_stage2 = dice_3d(pred_stage2, gt_zyx)
            write_mask_like(pred_stage2, img_sitk, out_mask_path)
            print(
                f"[OK] {patient_id}: stage1_dice={dice_stage1:.4f}, stage2_dice={dice_stage2:.4f}, "
                f"middle={middle_id} -> {out_mask_path}"
            )

            rows.append(
                {
                    "Patient_ID": patient_id,
                    "Prompt_Slice_ID": f"{upper_id},{lower_id},{middle_id}",
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Middle_ID": middle_id,
                    "Dice3D_Stage1": dice_stage1,
                    "Dice3D_Stage2": dice_stage2,
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    rows.sort(key=lambda r: int(re.search(r"(\d+)$", str(r["Patient_ID"])).group(1)) if re.search(r"(\d+)$", str(r["Patient_ID"])) else 10**9)

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["Patient_ID", "Prompt_Slice_ID", "Lower_Bound_ID", "Upper_Bound_ID", "Middle_ID", "Dice3D_Stage1", "Dice3D_Stage2"])
    for r in rows:
        ws.append(
            [
                r["Patient_ID"],
                r["Prompt_Slice_ID"],
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                int(r["Middle_ID"]),
                round(float(r["Dice3D_Stage1"]), 6),
                round(float(r["Dice3D_Stage2"]), 6),
            ]
        )
    wb.save(str(out_xlsx))
    print(f"[DONE] Excel saved: {out_xlsx}")
    print(f"[DONE] Masks saved in: {best_mask_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
