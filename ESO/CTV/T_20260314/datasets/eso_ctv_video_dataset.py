# 生成clip和prompt

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from ESO.CTV.T_20260314.utils.data_utils_ctv import VideoDatapoint, Frame, Object


class EsoCTVVideoDataset(torch.utils.data.Dataset):
    """
    最小版：
    - 每个病例返回一个 VideoDatapoint
    - 一个病例只包含 1 个 object（CTV）
    - 每层都放一个 object，mask 可为全0
    - 先不做复杂增强，只做窗宽窗位 + 三通道复制
    """

    def __init__(
            self,
            root_dir,
            image_name="image.nii.gz",
            mask_name="CTV.nii.gz",
            window_center=40,
            window_width=400,
            z_expand=0,
            max_frames=None,
            clip_len=8,
            stride=4,
            split="train",  # "train" 或 "val"
            num_folds=5,  # 五折
            fold_index=0,  # 当前 fold 索引 0~4
            seed=42,  # 随机种子
    ):
        self.root_dir = Path(root_dir)
        self.image_name = image_name
        self.mask_name = mask_name
        self.window_center = window_center
        self.window_width = window_width
        self.z_expand = z_expand
        self.max_frames = max_frames
        self.clip_len = clip_len
        self.stride = stride

        all_patient_dirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

        if not (0 <= fold_index < num_folds):
            raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}")

        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_patient_dirs))
        rng.shuffle(indices)

        folds = np.array_split(indices, num_folds)
        val_idx = set(folds[fold_index].tolist())

        if split == "train":
            selected_indices = [i for i in range(len(all_patient_dirs)) if i not in val_idx]
        elif split == "val":
            selected_indices = [i for i in range(len(all_patient_dirs)) if i in val_idx]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.patient_dirs = [all_patient_dirs[i] for i in selected_indices]

        self.samples = []

        for pdir in self.patient_dirs:
            img_path = pdir / self.image_name
            mask_path = pdir / self.mask_name

            if (not img_path.exists()) or (not mask_path.exists()):
                continue

            mask_zyx, _ = self._read_nii(mask_path)
            mask_zyx = (mask_zyx > 0).astype(np.uint8)

            pos = np.where(mask_zyx.sum(axis=(1, 2)) > 0)[0]
            if len(pos) == 0:
                continue

            z0 = max(0, int(pos[0]) - self.z_expand)
            z1 = min(mask_zyx.shape[0] - 1, int(pos[-1]) + self.z_expand)

            total_len = z1 - z0 + 1

            # 如果范围长度不超过 clip_len，就只保留一个 clip
            if total_len <= self.clip_len:
                self.samples.append({
                    "patient_dir": pdir,
                    "z_start": z0,
                    "z_end": z1,
                    "prompt_mode": "start",
                })
                self.samples.append({
                    "patient_dir": pdir,
                    "z_start": z0,
                    "z_end": z1,
                    "prompt_mode": "end",
                })
            else:
                # 先生成所有唯一 clip（不带 prompt_mode）
                clip_ranges = []

                start = z0
                while start + self.clip_len - 1 <= z1:
                    end = start + self.clip_len - 1
                    clip_ranges.append((start, end))
                    start += self.stride

                # 补一个以下界结尾的 clip，防止最后尾巴漏掉
                last_start = z1 - self.clip_len + 1
                last_end = z1
                if len(clip_ranges) == 0 or clip_ranges[-1] != (last_start, last_end):
                    clip_ranges.append((last_start, last_end))

                # 根据位置决定 prompt_mode
                # 第一个 clip：只 start
                # 最后一个 clip：只 end
                # 中间 clip：start + end
                for i, (clip_start, clip_end) in enumerate(clip_ranges):
                    if i == 0:
                        self.samples.append({
                            "patient_dir": pdir,
                            "z_start": clip_start,
                            "z_end": clip_end,
                            "prompt_mode": "start",
                        })
                    elif i == len(clip_ranges) - 1:
                        self.samples.append({
                            "patient_dir": pdir,
                            "z_start": clip_start,
                            "z_end": clip_end,
                            "prompt_mode": "end",
                        })
                    else:
                        self.samples.append({
                            "patient_dir": pdir,
                            "z_start": clip_start,
                            "z_end": clip_end,
                            "prompt_mode": "start",
                        })
                        self.samples.append({
                            "patient_dir": pdir,
                            "z_start": clip_start,
                            "z_end": clip_end,
                            "prompt_mode": "end",
                        })


    def __len__(self):
        return len(self.samples)

    def _read_nii(self, path):
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # [Z, H, W]
        return arr, img

    def _window_to_uint8(self, img2d):
        img = img2d.astype(np.float32)
        lo = self.window_center - self.window_width / 2.0
        hi = self.window_center + self.window_width / 2.0
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo + 1e-6)
        img = (img * 255.0).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdir = sample["patient_dir"]
        z_start = sample["z_start"]
        z_end = sample["z_end"]
        prompt_mode = sample["prompt_mode"]

        if prompt_mode == "start":
            prompt_frame_idx = 0
        elif prompt_mode == "end":
            prompt_frame_idx = z_end - z_start
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

        img_path = pdir / self.image_name
        mask_path = pdir / self.mask_name

        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        img_zyx, _ = self._read_nii(img_path)
        mask_zyx, _ = self._read_nii(mask_path)

        mask_zyx = (mask_zyx > 0).astype(np.uint8)

        img_clip = img_zyx[z_start:z_end + 1]
        mask_clip = mask_zyx[z_start:z_end + 1]

        frames = []
        for t in range(img_clip.shape[0]):
            # 图像转 3 通道 tensor [3, H, W]
            u8 = self._window_to_uint8(img_clip[t])
            rgb = np.stack([u8, u8, u8], axis=0)   # [3, H, W]
            image_tensor = torch.from_numpy(rgb).float() / 255.0

            # mask [H, W]
            mask_tensor = torch.from_numpy(mask_clip[t]).to(torch.bool)

            obj = Object(
                object_id=1,
                frame_index=t,
                segment=mask_tensor,
            )

            frame = Frame(
                data=image_tensor,
                objects=[obj],
            )
            frames.append(frame)

        # video_id 先用 idx
        video = VideoDatapoint(
            frames=frames,
            video_id=idx,
            size=(img_clip.shape[1], img_clip.shape[2]),
            prompt_frame_idx=prompt_frame_idx,
        )

        return video