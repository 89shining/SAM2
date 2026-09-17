from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


# =========================
# 1. 修改这里
# =========================
ROOT_DIR = Path(r"D:\WUSI\Prompt_mask\train")

POS_NAME = "pos_erode2_top3_min50mm2_dilate2.nii.gz"
NEG_NAME = "neg_erode2_top3_min50mm2_dilate2.nii.gz"

POS_OUT = "pos_skeleton.nii.gz"
NEG_OUT = "neg_skeleton.nii.gz"


def skeletonize_nii_2d(input_path: Path, output_path: Path):
    """
    对 3D NIfTI 的每个轴位 slice 独立执行 2D skeletonization。
    输出保持原始 affine/header。
    """

    nii = nib.load(str(input_path))
    data = nii.get_fdata()

    # 二值化
    binary = data > 0

    # 输出数组
    skeleton_volume = np.zeros(binary.shape, dtype=np.uint8)

    # 假设 NIfTI shape 为 (X, Y, Z)
    # 沿 Z 方向逐层做 2D skeletonization
    for z in range(binary.shape[2]):
        slice_mask = binary[:, :, z]

        if np.any(slice_mask):
            slice_skeleton = skeletonize(slice_mask)
            skeleton_volume[:, :, z] = slice_skeleton.astype(np.uint8)

    # 保存，同时继承空间信息
    header = nii.header.copy()
    header.set_data_dtype(np.uint8)

    out_nii = nib.Nifti1Image(
        skeleton_volume,
        affine=nii.affine,
        header=header
    )

    nib.save(out_nii, str(output_path))

    # 简单统计
    original_voxels = int(binary.sum())
    skeleton_voxels = int(skeleton_volume.sum())

    print(f"[OK] {input_path.name}")
    print(f"     original voxels : {original_voxels}")
    print(f"     skeleton voxels : {skeleton_voxels}")
    if original_voxels > 0:
        print(f"     retained ratio  : {skeleton_voxels / original_voxels:.4f}")
    print(f"     saved to        : {output_path}")


def main():

    patient_dirs = sorted(
        [p for p in ROOT_DIR.iterdir()
         if p.is_dir() and p.name.startswith("p_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    print(f"Found {len(patient_dirs)} patient folders.\n")

    for patient_dir in patient_dirs:

        print("=" * 70)
        print(f"Processing: {patient_dir.name}")

        # POS
        pos_path = patient_dir / POS_NAME
        pos_out_path = patient_dir / POS_OUT

        if pos_path.exists():
            skeletonize_nii_2d(pos_path, pos_out_path)
        else:
            print(f"[Missing POS] {pos_path}")

        # NEG
        neg_path = patient_dir / NEG_NAME
        neg_out_path = patient_dir / NEG_OUT

        if neg_path.exists():
            skeletonize_nii_2d(neg_path, neg_out_path)
        else:
            print(f"[Missing NEG] {neg_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()