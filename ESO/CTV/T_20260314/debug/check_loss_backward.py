import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from ESO.CTV.T_20260314.datasets.eso_ctv_train_data import EsoCTVTrainData


def main():
    # ===== 1. 准备一个 batch =====
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

    print("batch.img_batch shape:", batch.img_batch.shape)
    print("batch.masks shape:", batch.masks.shape)

    # ===== 2. 读取 config =====
    config_dir = r"D:\project\SAM2\ESO\CTV\T_20260314\configs"
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_ctv_finetune")

    # ===== 3. 实例化 model 和 loss =====
    model = instantiate(cfg.trainer.model, _convert_="all")
    loss_fn = instantiate(cfg.trainer.loss["all"], _convert_="all")

    model.train()

    # ===== 4. forward =====
    outputs = model(batch)

    # ===== 5. loss =====
    targets = batch.masks
    loss_dict = loss_fn(outputs, targets)

    print("loss_dict type:", type(loss_dict))
    print("loss_dict keys:", loss_dict.keys() if isinstance(loss_dict, dict) else "not dict")

    if isinstance(loss_dict, dict):
        loss = loss_dict["core_loss"]
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                print(f"{k}: {v.item():.6f}")
            else:
                print(f"{k}: {v}")
    else:
        loss = loss_dict
        print("loss:", loss.item())

    # ===== 6. backward =====
    model.zero_grad(set_to_none=True)
    loss.backward()

    # ===== 7. 检查是否真的有梯度 =====
    grad_count = 0
    grad_names = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_count += 1
            if len(grad_names) < 10:
                grad_names.append(name)

    print("backward success!")
    print("params with grad:", grad_count)
    print("example grad params:", grad_names)


if __name__ == "__main__":
    main()