import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from PIL import Image

def window_level(img, center=40, width=400):
    img = img.astype(np.float32)
    lo = center - width / 2
    hi = center + width / 2
    img = np.clip(img, lo, hi)
    img = (img - lo) / width * 255.0
    return img.astype(np.uint8)


class SAM2PatientSliceDataset(Dataset):
    """
    输入：若干 patient folder（p_x）
    输出：slice-level samples
    """

    def __init__(self, root_dir, patient_list):
        self.samples = []

        for pid in patient_list:
            p_dir = os.path.join(root_dir, pid)

            img_nii = sitk.ReadImage(os.path.join(p_dir, "image.nii.gz"))
            gt_nii = sitk.ReadImage(os.path.join(p_dir, "CTV.nii.gz"))
            prompt_nii = sitk.ReadImage(os.path.join(p_dir, "prompt.nii.gz"))

            img_arr = sitk.GetArrayFromImage(img_nii)        # (Z, H, W)
            gt_arr = sitk.GetArrayFromImage(gt_nii)
            prompt_arr = sitk.GetArrayFromImage(prompt_nii)

            Z = img_arr.shape[0]

            for z in range(Z):
                if gt_arr[z].sum() == 0:
                    continue   # ✅ 只用 GT 非空切片

                self.samples.append({
                    "img": img_arr[z],
                    "gt": gt_arr[z],
                    "prompt": prompt_arr[z],
                    "pid": pid,
                    "z": z
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = window_level(s["img"])
        img = Image.fromarray(img).resize((1024, 1024))
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0).repeat(3, 1, 1)

        # nnUNet mask → 256×256 prompt
        pm = (s["prompt"] > 0).astype(np.uint8)
        pm = Image.fromarray(pm * 255).resize((256, 256), Image.NEAREST)
        pm = torch.from_numpy(np.array(pm) > 0).float().unsqueeze(0)

        gt = torch.from_numpy(s["gt"]).float().unsqueeze(0)

        return {
            "image": img,
            "mask_prompt": pm,
            "gt": gt
        }
