import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import csv
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from ESO.CTV.T_20260314.utils.data_utils_ctv import (
    Frame, Object, VideoDatapoint, collate_fn
)


# ---------------------- 工具函数 ----------------------
def read_nii(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, img


def window_to_uint8(img2d, wc=40, ww=400):
    img = np.clip(img2d, wc - ww / 2, wc + ww / 2)
    img = ((img - (wc - ww / 2)) / ww * 255).astype(np.uint8)
    return img


def save_pred_nii(pred_arr, ref_img, save_path):
    pred_img = sitk.GetImageFromArray(pred_arr.astype(np.uint8))
    pred_img.CopyInformation(ref_img)
    sitk.WriteImage(pred_img, str(save_path))


# ---------------------- 测试数据集 ----------------------
class TestDataset:
    def __init__(self, root_dir, image_name="image.nii.gz", mask_name="CTV.nii.gz", clip_len=8):
        self.root_dir = Path(root_dir)
        self.image_name = image_name
        self.mask_name = mask_name
        self.clip_len = clip_len
        self.patients = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pdir = self.patients[idx]
        img_path = pdir / self.image_name
        mask_path = pdir / self.mask_name

        img_zyx, img_sitk = read_nii(img_path)
        mask_zyx, _ = read_nii(mask_path)
        mask_zyx = (mask_zyx > 0).astype(np.uint8)

        pos = np.where(mask_zyx.sum(axis=(1, 2)) > 0)[0]
        if len(pos) == 0:
            raise ValueError(f"No positive mask found in {pdir}")

        z0, z1 = int(pos[0]), int(pos[-1])

        num_frames = z1 - z0 + 1
        if num_frames <= self.clip_len:
            clip_ranges = [(z0, z1)]
            prompt_layers = [z0]   # 绝对层号
        else:
            clip_ranges = []
            start = z0
            while start + self.clip_len - 1 <= z1:
                clip_ranges.append((start, start + self.clip_len - 1))
                start += self.clip_len

            if clip_ranges[-1][1] < z1:
                clip_ranges.append((z1 - self.clip_len + 1, z1))

            prompt_layers = [s + (e - s) // 2 for s, e in clip_ranges]  # 绝对层号

        return {
            "patient_dir": pdir,
            "img_zyx": img_zyx,
            "img_sitk": img_sitk,
            "mask_zyx": mask_zyx,
            "clip_ranges": clip_ranges,
            "prompt_layers": prompt_layers,
        }


# ---------------------- 构建 clip ----------------------
def build_clip(img_zyx, mask_zyx, start, end, prompt_frame_idx_in_clip):
    frames = []
    for local_t, z in enumerate(range(start, end + 1)):
        u8 = window_to_uint8(img_zyx[z])
        rgb = np.stack([u8, u8, u8], axis=0)  # [3,H,W]
        image_tensor = torch.from_numpy(rgb).float() / 255.0

        mask_tensor = torch.from_numpy(mask_zyx[z]).to(torch.bool)

        obj = Object(
            object_id=1,
            frame_index=local_t,   # 用 clip 内索引更稳妥
            segment=mask_tensor
        )
        frame = Frame(data=image_tensor, objects=[obj])
        frames.append(frame)

    video = VideoDatapoint(
        frames=frames,
        video_id=0,
        size=(img_zyx.shape[1], img_zyx.shape[2]),  # (H, W)
        prompt_frame_idx=int(prompt_frame_idx_in_clip),
    )
    return video


# ---------------------- 测试主函数 ----------------------
def test_model(
    test_root,
    ckpt_path,
    save_root,
    config_dir,
    config_name="sam2_ctv_finetune",
    device="cuda",
    clip_len=8,
    fusion_mode="max",   # "max" or "mean"
    threshold=0.0,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1) 用和训练相同的配置实例化模型
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg.trainer.model, _convert_="all")
    state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()

    dataset = TestDataset(test_root, clip_len=clip_len)

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    prompt_log_path = save_root / "prompt_layers_info.csv"
    with open(prompt_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "prompt_layers_abs", "num_prompts"])

        for data in tqdm(dataset, desc="Patients"):
            img_zyx = data["img_zyx"]
            mask_zyx = data["mask_zyx"]
            merged_pred = np.zeros_like(mask_zyx, dtype=np.float32)
            count_map = np.zeros_like(mask_zyx, dtype=np.float32)

            patient_name = data["patient_dir"].name
            writer.writerow([patient_name, data["prompt_layers"], len(data["prompt_layers"])])

            for s, e in data["clip_ranges"]:
                # 找到落在当前 clip 内的绝对提示层
                prompt_idx_abs = [p for p in data["prompt_layers"] if s <= p <= e]

                if len(prompt_idx_abs) == 0:
                    prompt_frame_idx_in_clip = 0
                else:
                    prompt_frame_idx_in_clip = prompt_idx_abs[0] - s

                clip = build_clip(
                    img_zyx=img_zyx,
                    mask_zyx=mask_zyx,
                    start=s,
                    end=e,
                    prompt_frame_idx_in_clip=prompt_frame_idx_in_clip,
                )

                # 2) 变成 BatchedVideoDatapoint
                batch = collate_fn([clip], dict_key="all")
                batch = batch.to(device)

                with torch.no_grad():
                    out = model(batch)

                    # out: 长度为 T 的 list；每个元素里 pred_masks_high_res 是 [B,1,H,W]
                    pred_clip = torch.cat([frame_out["pred_masks_high_res"] for frame_out in out], dim=0)  # [T,1,H,W]
                    pred_clip = pred_clip[:, 0].cpu().numpy()  # [T,H,W]

                    # 二值化
                    pred_clip = (pred_clip > threshold).astype(np.float32)

                    if fusion_mode == "max":
                        merged_pred[s:e + 1] = np.maximum(merged_pred[s:e + 1], pred_clip)
                    elif fusion_mode == "mean":
                        merged_pred[s:e + 1] += pred_clip
                        count_map[s:e + 1] += 1
                    else:
                        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

            if fusion_mode == "mean":
                merged_pred = merged_pred / np.maximum(count_map, 1)

            merged_pred = (merged_pred > 0.5).astype(np.uint8)

            patient_idx = dataset.patients.index(data["patient_dir"])
            save_path = save_root / f"CTV_{patient_idx:03d}.nii.gz"
            save_pred_nii(merged_pred, data["img_sitk"], save_path)

    print(f"All patients done. Prompt layers info saved to {prompt_log_path}")


if __name__ == "__main__":
    test_model(
        test_root="/home/wusi/segment-anything/SAMdata/Eso/20251217_CTV/datanii/test_nii",
        ckpt_path="/home/wusi/SAM2/SAM2data/20260315/TrainResult/fold_0/checkpoints/best.pth",
        save_root="/home/wusi/SAM2/SAM2data/20260315/TestResult",
        config_dir="/home/wusi/SAM2/ESO/CTV/T_20260314/configs",
        config_name="sam2_ctv_finetune",
        device="cuda",
        clip_len=8,
        fusion_mode="max",
        threshold=0.0,
    )