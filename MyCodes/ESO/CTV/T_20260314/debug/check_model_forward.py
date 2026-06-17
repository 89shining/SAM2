import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from MyCodes.ESO.CTV.T_20260314.datasets.eso_ctv_train_data import EsoCTVTrainData


def main():
    # ===== 1. 先准备一个 batch =====
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
    from MyCodes.ESO.CTV.T_20260314.utils.data_utils_ctv import collate_fn

    dataset = data.dataset
    batch = collate_fn([dataset[1]], dict_key="all")

    print("batch.img_batch shape:", batch.img_batch.shape)
    print("batch.masks shape:", batch.masks.shape)
    print("batch.metadata.prompt_frame_idx shape:", batch.metadata.prompt_frame_idx.shape)
    print("batch.metadata.prompt_frame_idx:", batch.metadata.prompt_frame_idx[:, 0])

    # ===== 2. 从你自己的 config 文件夹读取 yaml =====
    config_dir = r"/MyCodes/ESO/CTV/T_20260314/configs"

    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_ctv_finetune")

    # ===== 3. 实例化 model =====
    model = instantiate(cfg.trainer.model, _convert_="all")
    model.eval()

    # ===== 4. 单次 forward =====
    with torch.no_grad():
        outputs = model(batch)

    print("forward success!")
    print("len(outputs):", len(outputs))
    print("first output keys:", outputs[0].keys())

    if "pred_masks_high_res" in outputs[0]:
        print("first pred_masks_high_res shape:", outputs[0]["pred_masks_high_res"].shape)
    if "pred_masks" in outputs[0]:
        print("first pred_masks shape:", outputs[0]["pred_masks"].shape)


if __name__ == "__main__":
    main()