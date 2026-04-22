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
PROJECT_ROOT = Path(__file__).resolve().parents[7]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2.build_sam import build_sam2_video_predictor


# ================= Defaults =================
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


def _safe_hd95_2d(pred2d: np.ndarray, gt2d: np.ndarray) -> float:
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
        score = _safe_hd95_2d(pred_zyx[z], gt_zyx[z])
        if score >= 0:
            valid_scores.append((z, score))
    if len(valid_scores) == 0:
        return int(lower_id)
    return int(max(valid_scores, key=lambda x: x[1])[0])


def relative_pos_upper_to_lower(slice_id: int, lower_id: int, upper_id: int):
    if upper_id == lower_id:
        return 0.0
    return float((upper_id - slice_id) / (upper_id - lower_id))


@torch.no_grad()
def infer_selector_middle_hd95(
    selector_predictor,
    frame_dir: Path,
    gt_zyx: np.ndarray,
    lower_id: int,
    upper_id: int,
    obj_id: int,
):
    state = selector_predictor.init_state(video_path=str(frame_dir))
    selector_predictor.reset_state(state)

    stage1_prompt_ids = [int(upper_id)]
    if int(lower_id) != int(upper_id):
        stage1_prompt_ids.append(int(lower_id))
    stage1_prompt_ids = list(dict.fromkeys(stage1_prompt_ids))
    for sid in stage1_prompt_ids:
        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")
        selector_predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=obj_id,
            mask=prompt_mask,
        )

    z, h, w = gt_zyx.shape
    pred_stage1 = np.zeros((z, h, w), dtype=np.uint8)
    for fidx, obj_ids, logits in selector_predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == obj_id:
                pred_stage1[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break

    middle_id = select_middle_from_pred_and_gt_hd95(
        pred_zyx=pred_stage1,
        gt_zyx=gt_zyx,
        lower_id=lower_id,
        upper_id=upper_id,
    )
    return pred_stage1, int(middle_id)


@torch.no_grad()
def infer_two_stage_iterative_with_fixed_middle(
    main_predictor,
    frame_dir: Path,
    gt_zyx: np.ndarray,
    lower_id: int,
    upper_id: int,
    middle_id: int,
    obj_id: int,
):
    state = main_predictor.init_state(video_path=str(frame_dir))
    main_predictor.reset_state(state)

    # Stage-1: upper/lower only
    stage1_prompt_ids = [int(upper_id)]
    if int(lower_id) != int(upper_id):
        stage1_prompt_ids.append(int(lower_id))
    stage1_prompt_ids = list(dict.fromkeys(stage1_prompt_ids))
    for sid in stage1_prompt_ids:
        prompt_mask = (gt_zyx[sid] > 0).astype(np.uint8)
        if prompt_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {sid} is empty in GT.")
        main_predictor.add_new_mask(
            inference_state=state,
            frame_idx=sid,
            obj_id=obj_id,
            mask=prompt_mask,
        )

    z, h, w = gt_zyx.shape
    pred_stage1 = np.zeros((z, h, w), dtype=np.uint8)
    for fidx, obj_ids, logits in main_predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == obj_id:
                pred_stage1[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break

    # Stage-2: inherit stage-1 memory and inject middle prompt
    if int(middle_id) not in stage1_prompt_ids:
        mid_mask = (gt_zyx[int(middle_id)] > 0).astype(np.uint8)
        if mid_mask.sum() == 0:
            raise RuntimeError(f"Prompt slice {middle_id} is empty in GT.")
        main_predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(middle_id),
            obj_id=obj_id,
            mask=mid_mask,
        )

    pred_stage2 = np.zeros((z, h, w), dtype=np.uint8)
    for fidx, obj_ids, logits in main_predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            if int(oid) == obj_id:
                pred_stage2[int(fidx)] = (logits[i] > 0).cpu().numpy()
                break
    return pred_stage1, pred_stage2


def patient_id_from_folder(pdir: Path):
    m = re.search(r"\d+", pdir.name)
    if m is None:
        raise ValueError(f"Cannot parse patient id from folder name: {pdir.name}")
    return f"CTV_{int(m.group()):03d}"


def resolve_ckpt(finetuned_ckpt, train_output_root: Path) -> Path:
    """
    Priority:
    1) best fold from best_fold.txt under train_output_root
    2) explicit --finetuned-ckpt if provided and exists
    """
    best_fold_txt = train_output_root / "best_fold.txt"
    if best_fold_txt.exists():
        content = best_fold_txt.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"best_ckpt:\s*(.+)", content)
        if m:
            best_ckpt = Path(m.group(1).strip())
            if best_ckpt.exists():
                return best_ckpt

    if finetuned_ckpt is not None:
        finetuned_ckpt = Path(finetuned_ckpt)
        if finetuned_ckpt.exists():
            return finetuned_ckpt
        raise FileNotFoundError(f"--finetuned-ckpt not found: {finetuned_ckpt}")

    raise FileNotFoundError("No usable finetuned checkpoint found. Tried best_fold.txt and optional --finetuned-ckpt.")


def main():
    parser = argparse.ArgumentParser("Test SAM2 with fixed external-HD95 middle prompts")
    parser.add_argument("--test-root", type=Path, required=True, help="Separate test set root")
    parser.add_argument("--output-root", type=Path, required=True, help="Save root for masks/excel")
    parser.add_argument("--finetuned-ckpt", type=Path, default=None, help="Optional checkpoint for inference fallback")
    parser.add_argument("--train-output-root", type=Path, required=True, help="Training output root for auto resolving best fold ckpt")
    parser.add_argument("--selector-train-output-root", type=Path, required=True, help="Selector training output root for fixed HD95-middle checkpoint")
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--img-name", type=str, default="image.nii.gz")
    parser.add_argument("--gt-name", type=str, default="CTV.nii.gz")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--excel-name", type=str, default="prompt_layer_search_fixed_external_hd95_middle.xlsx")
    args = parser.parse_args()

    if not args.test_root.exists():
        raise FileNotFoundError(f"test root not found: {args.test_root}")
    ckpt_path = resolve_ckpt(args.finetuned_ckpt, args.train_output_root)
    selector_ckpt_path = resolve_ckpt(None, args.selector_train_output_root)

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
    selector_predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(selector_ckpt_path),
        device=device,
    )
    print(f"[INFO] Using checkpoint: {ckpt_path}")
    print(f"[INFO] Using selector checkpoint: {selector_ckpt_path}")

    patient_dirs = sorted([p for p in args.test_root.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(patient_dirs)} patients")

    all_rows = []
    best_rows = []

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
        middle_id = lower_id

        rel = relative_pos_upper_to_lower(middle_id, lower_id, upper_id)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_test_{pdir.name}_"))
        try:
            save_frames_from_volume(img_zyx, tmp_dir, args.window_center, args.window_width)
            _, middle_id = infer_selector_middle_hd95(
                selector_predictor=selector_predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                obj_id=args.obj_id,
            )
            rel = relative_pos_upper_to_lower(middle_id, lower_id, upper_id)
            print(
                f"[INFO] {patient_id} | fixed prompts: upper={upper_id}, lower={lower_id}, middle={middle_id}"
            )

            pred_stage1, pred_stage2 = infer_two_stage_iterative_with_fixed_middle(
                main_predictor=predictor,
                frame_dir=tmp_dir,
                gt_zyx=gt_zyx,
                lower_id=lower_id,
                upper_id=upper_id,
                middle_id=middle_id,
                obj_id=args.obj_id,
            )
            dice_stage1 = dice_3d(pred_stage1, gt_zyx)
            dice_stage2 = dice_3d(pred_stage2, gt_zyx)
            write_mask_like(pred_stage2, img_sitk, out_mask_path)
            print(
                f"[OK] {patient_id}: stage1_dice={dice_stage1:.4f}, "
                f"stage2_dice={dice_stage2:.4f} -> {out_mask_path}"
            )

            all_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Prompt_Slice_ID": f"{upper_id},{lower_id},{middle_id}",
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Prompt_Rel_Pos_UpperToLower": rel,
                    "Dice3D_Stage1": dice_stage1,
                    "Dice3D_Stage2": dice_stage2,
                }
            )
            best_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Best_Prompt_Slice_ID": middle_id,
                    "Lower_Bound_ID": lower_id,
                    "Upper_Bound_ID": upper_id,
                    "Best_Prompt_Rel_Pos_UpperToLower": rel,
                    "Best_Dice3D_Stage1": dice_stage1,
                    "Best_Dice3D_Stage2": dice_stage2,
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _pid_key(row):
        mm = re.search(r"(\d+)$", str(row["Patient_ID"]))
        return int(mm.group(1)) if mm else 10**9

    all_rows.sort(key=_pid_key)
    best_rows.sort(key=_pid_key)

    wb = Workbook()
    ws_all = wb.active
    ws_all.title = "All_Prompt_Search"
    ws_all.append(
        [
            "Patient_ID",
            "Prompt_Slice_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Prompt_Rel_Pos_UpperToLower",
            "Dice3D_Stage1",
            "Dice3D_Stage2",
        ]
    )
    for r in all_rows:
        ws_all.append(
            [
                r["Patient_ID"],
                r["Prompt_Slice_ID"],
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                round(float(r["Prompt_Rel_Pos_UpperToLower"]), 6),
                round(float(r["Dice3D_Stage1"]), 6),
                round(float(r["Dice3D_Stage2"]), 6),
            ]
        )

    ws_best = wb.create_sheet("Best_Per_Patient")
    ws_best.append(
        [
            "Patient_ID",
            "Best_Prompt_Slice_ID",
            "Lower_Bound_ID",
            "Upper_Bound_ID",
            "Best_Prompt_Rel_Pos_UpperToLower",
            "Best_Dice3D_Stage1",
            "Best_Dice3D_Stage2",
        ]
    )
    for r in best_rows:
        ws_best.append(
            [
                r["Patient_ID"],
                int(r["Best_Prompt_Slice_ID"]),
                int(r["Lower_Bound_ID"]),
                int(r["Upper_Bound_ID"]),
                round(float(r["Best_Prompt_Rel_Pos_UpperToLower"]), 6),
                round(float(r["Best_Dice3D_Stage1"]), 6),
                round(float(r["Best_Dice3D_Stage2"]), 6),
            ]
        )

    wb.save(str(out_xlsx))
    print(f"[DONE] Excel saved: {out_xlsx}")
    print(f"[DONE] Masks saved in: {best_mask_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
