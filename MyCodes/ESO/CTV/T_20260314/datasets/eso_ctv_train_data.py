from torch.utils.data import DataLoader

from MyCodes.ESO.CTV.T_20260314.utils.data_utils_ctv import collate_fn
from MyCodes.ESO.CTV.T_20260314.datasets.eso_ctv_video_dataset import EsoCTVVideoDataset


class EsoCTVTrainData:
    """
    给 Trainer 用的最小数据包装器：
    Trainer 会调用 get_loader(epoch)，这里返回 DataLoader。
    """

    def __init__(
            self,
            root_dir,
            batch_size=1,
            num_workers=0,
            shuffle=True,
            drop_last=False,
            dict_key="all",
            image_name="image.nii.gz",
            mask_name="CTV.nii.gz",
            window_center=40,
            window_width=400,
            z_expand=0,
            clip_len=8,
            stride=4,
            split="train",  # 新增
            fold_index=0,  # 新增
            num_folds=5,  # 新增
            seed=42,  # 新增
    ):
        self.dataset = EsoCTVVideoDataset(
            root_dir=root_dir,
            image_name=image_name,
            mask_name=mask_name,
            window_center=window_center,
            window_width=window_width,
            z_expand=z_expand,
            clip_len=clip_len,
            stride=stride,
            split=split,  # 新增参数，"train" 或 "val"
            fold_index=fold_index,  # 当前折
            num_folds=num_folds,  # 总折数
            seed=seed,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dict_key = dict_key

    def get_loader(self, epoch=0):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.drop_last,
            collate_fn=lambda batch: collate_fn(batch, dict_key=self.dict_key),
        )