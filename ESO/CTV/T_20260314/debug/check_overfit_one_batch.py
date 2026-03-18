import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from ESO.CTV.T_20260314.datasets.eso_ctv_train_data import EsoCTVTrainData


def main():
    # 1. 一个固定 batch
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
        z_expand=2,
        max_frames=5,
    )
    loader = data.get_loader(epoch=0)
    batch = next(iter(loader))

    # 2. 配置
    config_dir = r"D:\project\SAM2\ESO\CTV\T_20260314\configs"
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_ctv_finetune")

    # 3. model + loss + optimizer
    model = instantiate(cfg.trainer.model, _convert_="all")
    loss_fn = instantiate(cfg.trainer.loss["all"], _convert_="all")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

    model.train()

    # 4. 反复训练同一个 batch
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch)
        loss_dict = loss_fn(outputs, batch.masks)
        loss = loss_dict["core_loss"]

        loss.backward()
        optimizer.step()

        print(
            f"step {step:02d} | "
            f"core_loss={loss_dict['core_loss'].item():.4f} | "
            f"mask={loss_dict['loss_mask'].item():.4f} | "
            f"dice={loss_dict['loss_dice'].item():.4f} | "
            f"iou={loss_dict['loss_iou'].item():.4f} | "
            f"class={loss_dict['loss_class'].item():.4f}"
        )


if __name__ == "__main__":
    main()