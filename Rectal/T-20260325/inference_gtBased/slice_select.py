"""
挑选上下界层和中间dice最差的层
"""

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_nii(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [Z, H, W]
    return arr


def binarize(arr):
    return (arr > 0).astype(np.uint8)


def dice2d(a, b, eps=1e-8):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    sa = a.sum()
    sb = b.sum()
    if sa == 0 and sb == 0:
        return 1.0
    return (2.0 * inter) / (sa + sb + eps)


def get_nonzero_slices(mask_3d):
    return [z for z in range(mask_3d.shape[0]) if mask_3d[z].sum() > 0]


def find_first_last_nonzero(mask_3d):
    nz = get_nonzero_slices(mask_3d)
    if len(nz) == 0:
        return None, None
    return nz[0], nz[-1]


def find_boundary_prompt_slices(gt):
    """
    两层提示：直接取 GT 的上下界层
    """
    z_gt_min, z_gt_max = find_first_last_nonzero(gt)
    if z_gt_min is None:
        raise ValueError("GT 掩膜为空。")
    return z_gt_min, z_gt_max


def find_middle_failure_slice(gt, pred, upper_prompt, lower_prompt,
                              exclude_gt_boundary_neighbors=1,
                              exclude_mismatch_neighbors=1):
    """
    三层提示中的中间层：
    在排除边界层、边界邻层、以及空非空不一致层及其邻层后，
    在剩余 GT 非空层中选择 Dice 最低的一层。
    """
    z_gt_min, z_gt_max = find_first_last_nonzero(gt)
    z_pr_min, z_pr_max = find_first_last_nonzero(pred)

    if z_gt_min is None:
        raise ValueError("GT 掩膜为空。")

    # GT/PRED 联合范围，用来找 mismatch
    valid_starts = [z for z in [z_gt_min, z_pr_min] if z is not None]
    valid_ends = [z for z in [z_gt_max, z_pr_max] if z is not None]
    z_start = min(valid_starts)
    z_end = max(valid_ends)

    exclude_slices = set()

    # 1) 排除 GT 上下界层及其邻层
    for z in [upper_prompt, lower_prompt]:
        for dz in range(-exclude_gt_boundary_neighbors, exclude_gt_boundary_neighbors + 1):
            zz = z + dz
            if 0 <= zz < gt.shape[0]:
                exclude_slices.add(zz)

    # 2) 排除 GT/PRED 空非空不一致层及其邻层
    mismatch_slices = []
    for z in range(z_start, z_end + 1):
        gt_nonempty = gt[z].sum() > 0
        pred_nonempty = pred[z].sum() > 0
        if gt_nonempty != pred_nonempty:
            mismatch_slices.append(z)

    for z in mismatch_slices:
        for dz in range(-exclude_mismatch_neighbors, exclude_mismatch_neighbors + 1):
            zz = z + dz
            if 0 <= zz < gt.shape[0]:
                exclude_slices.add(zz)

    # 3) 在剩余 GT 非空层中找 Dice 最低的一层
    candidate_slices = []
    for z in range(z_gt_min, z_gt_max + 1):
        if z in exclude_slices:
            continue
        if gt[z].sum() == 0:
            continue
        d = dice2d(gt[z], pred[z])
        candidate_slices.append((z, d))

    if len(candidate_slices) == 0:
        return None

    middle_prompt = min(candidate_slices, key=lambda x: x[1])[0]
    return middle_prompt


def compute_prompt_slices_for_case(gt_path, pred_path):
    gt = binarize(read_nii(gt_path))
    pred = binarize(read_nii(pred_path))

    if gt.shape != pred.shape:
        raise ValueError(f"shape 不一致: gt={gt.shape}, pred={pred.shape}")

    # 两层提示：GT 上下界
    upper_prompt, lower_prompt = find_boundary_prompt_slices(gt)

    # 三层提示：GT 上下界 + 纯中间失败层
    middle_prompt = find_middle_failure_slice(
        gt,
        pred,
        upper_prompt=upper_prompt,
        lower_prompt=lower_prompt,
        exclude_gt_boundary_neighbors=1,
        exclude_mismatch_neighbors=1,
    )

    result = {
        "two_prompt_z1": upper_prompt,
        "two_prompt_z2": lower_prompt,
        "three_prompt_z1": upper_prompt,
        "three_prompt_z2": middle_prompt,
        "three_prompt_z3": lower_prompt,
    }
    return result


def main():
    gt_dir = Path(r"/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset011_RectalCTV146p/labelsTs")
    pred_dir = Path(r"/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset011_RectalCTV146p/nnUNetTrainer__nnUNetPlans__3d_fullres/testResult_fold0")
    out_csv = Path(r"/home/wusi/SAM2/SAM2data/Rectal/20260325_CTV/prompt_slices.csv")

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if len(gt_files) == 0:
        raise FileNotFoundError(f"GT 目录未找到 .nii.gz 文件: {gt_dir}")

    rows = []
    for gt_path in gt_files:
        case_name = gt_path.name
        pred_path = pred_dir / case_name

        if not pred_path.exists():
            print(f"[Warning] 预测文件不存在，跳过: {case_name}")
            continue

        try:
            result = compute_prompt_slices_for_case(gt_path, pred_path)
            row = {"case_name": case_name, **result}
            rows.append(row)
            print(f"[OK] {case_name}: {row}")
        except Exception as e:
            print(f"[Error] {case_name}: {e}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "two_prompt_z1",
                "two_prompt_z2",
                "three_prompt_z1",
                "three_prompt_z2",
                "three_prompt_z3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n已保存到: {out_csv}")


if __name__ == "__main__":
    main()