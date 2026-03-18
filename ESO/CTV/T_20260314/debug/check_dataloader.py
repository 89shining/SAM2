# from ESO.CTV.T_20260314.datasets.eso_ctv_video_dataset import EsoCTVVideoDataset
#
# ds = EsoCTVVideoDataset(
#     root_dir=r"D:\SAM\Esophagus\20251217\datanii\train_nii",
#     z_expand=0,
#     clip_len=8,
#     stride=4,
# )
#
# print("num samples:", len(ds))
# for i in range(min(10, len(ds.samples))):
#     s = ds.samples[i]
#     print(i, s["patient_dir"].name, s["z_start"], s["z_end"], s["prompt_mode"])
#
# video = ds[0]
# print("video.prompt_frame_idx:", video.prompt_frame_idx)
# print("num frames in video:", len(video.frames))

# batch
from ESO.CTV.T_20260314.datasets.eso_ctv_train_data import EsoCTVTrainData

def main():
    data = EsoCTVTrainData(
        root_dir=r"D:\SAM\Esophagus\20251217\datanii\train_nii",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        dict_key="all",
        image_name="image.nii.gz",
        mask_name="CTV.nii.gz",
        window_center=40,
        window_width=400,
        z_expand=0,
        clip_len=8,
        stride=4,
    )

    loader = data.get_loader(epoch=0)
    batch = next(iter(loader))

    print("img_batch shape:", batch.img_batch.shape)
    print("masks shape:", batch.masks.shape)
    print("metadata prompt_frame_idx shape:", batch.metadata.prompt_frame_idx.shape)
    print("metadata prompt_frame_idx:", batch.metadata.prompt_frame_idx[:, 0])

if __name__ == "__main__":
    main()