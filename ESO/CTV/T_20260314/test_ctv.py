import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # /home/wusi/SAM2
sys.path.insert(0, str(PROJECT_ROOT))


import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ESO.CTV.T_20260314.utils.data_utils_ctv import Frame, Object, VideoDatapoint
from hydra.utils import instantiate
import csv


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
        self.patients = [p for p in self.root_dir.iterdir() if p.is_dir()]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pdir = self.patients[idx]
        img_path = pdir / self.image_name
        mask_path = pdir / self.mask_name
        img_zyx, img_sitk = read_nii(img_path)
        mask_zyx, _ = read_nii(mask_path)
        mask_zyx = (mask_zyx > 0).astype(np.uint8)

        # GT 上下界
        pos = np.where(mask_zyx.sum(axis=(1, 2)) > 0)[0]
        z0, z1 = pos[0], pos[-1]

        # 自动选择提示层和 clip
        num_frames = z1 - z0 + 1
        if num_frames <= self.clip_len:
            prompt_layers = [z0]
            clip_ranges = [(z0, z1)]
        else:
            clip_ranges = []
            start = z0
            while start + self.clip_len - 1 <= z1:
                clip_ranges.append((start, start + self.clip_len - 1))
                start += self.clip_len
            if clip_ranges[-1][1] < z1:
                clip_ranges.append((z1 - self.clip_len + 1, z1))
            prompt_layers = [s + (e - s) // 2 for s, e in clip_ranges]

        return {
            "patient_dir": pdir,
            "img_zyx": img_zyx,
            "img_sitk": img_sitk,
            "mask_zyx": mask_zyx,
            "clip_ranges": clip_ranges,
            "prompt_layers": prompt_layers
        }


# ---------------------- 构建 clip ----------------------
def build_clip(img_zyx, mask_zyx, start, end):
    frames = []
    for t in range(start, end + 1):
        u8 = window_to_uint8(img_zyx[t])
        rgb = np.stack([u8, u8, u8], axis=0)
        image_tensor = torch.from_numpy(rgb).float() / 255.0
        mask_tensor = torch.from_numpy(mask_zyx[t]).to(torch.bool)
        obj = Object(object_id=1, frame_index=t, segment=mask_tensor)
        frame = Frame(data=image_tensor, objects=[obj])
        frames.append(frame)
    return VideoDatapoint(frames=frames, video_id=0, size=(img_zyx.shape[2], img_zyx.shape[1]), prompt_frame_idx=0)


# ---------------------- 测试主函数 ----------------------
def test_model(
        test_root, ckpt_path, save_root, model_class, device="cuda",
        clip_len=8, fusion_mode="max"  # fusion_mode: "max" or "mean"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = model_class.to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = TestDataset(test_root, clip_len=clip_len)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # 保存提示信息 CSV
    prompt_log_path = save_root / "prompt_layers_info.csv"
    with open(prompt_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "prompt_layers", "num_prompts"])

        for data in tqdm(dataset, desc="Patients"):
            img_zyx = data["img_zyx"]
            mask_zyx = data["mask_zyx"]
            merged_pred = np.zeros_like(mask_zyx, dtype=np.float32)
            count_map = np.zeros_like(mask_zyx, dtype=np.float32)  # mean 融合用

            # 记录提示层
            patient_name = data["patient_dir"].name
            writer.writerow([patient_name, data["prompt_layers"], len(data["prompt_layers"])])

            for s, e in data["clip_ranges"]:
                prompt_idx = [p for p in data["prompt_layers"] if s <= p <= e]
                if len(prompt_idx) == 0:
                    prompt_idx = [s]

                clip = build_clip(img_zyx, mask_zyx, s, e)
                clip.prompt_frame_idx = prompt_idx[0]

                with torch.no_grad():
                    out = model(clip)
                    pred_clip = torch.cat([f["pred_masks_high_res"] for f in out], dim=0)
                    pred_clip = pred_clip[:, 0, 0].cpu().numpy()

                    if fusion_mode == "max":
                        merged_pred[s:e + 1] = np.maximum(merged_pred[s:e + 1], pred_clip)
                    elif fusion_mode == "mean":
                        merged_pred[s:e + 1] += pred_clip
                        count_map[s:e + 1] += 1
                    else:
                        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

            if fusion_mode == "mean":
                merged_pred = merged_pred / np.maximum(count_map, 1)

            patient_idx = dataset.patients.index(data["patient_dir"])
            save_path = save_root / f"CTV_{patient_idx:03d}.nii.gz"
            save_pred_nii(merged_pred, data["img_sitk"], save_path)

    print(f"All patients done. Prompt layers info saved to {prompt_log_path}")


# ---------------------- 使用示例 ----------------------
from ESO.CTV.T_20260314.modeling.sam2_train_ctv import SAM2TrainCTV
test_model(
    test_root="/home/wusi/SAMdata/Eso/20251217_CTV/datanii/test_nii",
    ckpt_path="/home/wusi/SAM2/SAM2data/20260315/TrainResult/fold_0/checkpoints/best.pth",
    save_root="/home/wusi/SAM2/SAM2data/20260315/TestResult",
    model_class=SAM2TrainCTV,
    device="cuda",
    clip_len=8,
    fusion_mode="max"
)