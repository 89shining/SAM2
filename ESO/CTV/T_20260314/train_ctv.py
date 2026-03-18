import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # /home/wusi/SAM2
sys.path.insert(0, str(PROJECT_ROOT))


import os
import csv
import torch
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter


def log_message(log_file, msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def save_checkpoint(model, optimizer, epoch, save_path):
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        save_path,
    )


def compute_batch_dice(pred_logits, gt_mask, threshold=0.0, eps=1e-6):
    """
    pred_logits: [B, 1, H, W] or [B, H, W]
    gt_mask:     [B, 1, H, W] or [B, H, W]
    """
    if pred_logits.dim() == 4 and pred_logits.shape[1] == 1:
        pred_logits = pred_logits[:, 0]
    if gt_mask.dim() == 4 and gt_mask.shape[1] == 1:
        gt_mask = gt_mask[:, 0]

    pred_bin = (pred_logits > threshold).float()
    gt_mask = gt_mask.float()

    inter = (pred_bin * gt_mask).sum(dim=(1, 2))
    union = pred_bin.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (union + eps)
    return dice.mean().item()

def build_optimizer(model, cfg):
    optimizer_type = cfg.optimizer_schemes[cfg.scratch.optimizer].type

    image_encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("image_encoder"):
            image_encoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = []

    if len(other_params) > 0:
        param_groups.append(
            {
                "params": other_params,
                "lr": cfg.scratch.base_lr,
                "weight_decay": cfg.scratch.weight_decay,
            }
        )

    if len(image_encoder_params) > 0:
        param_groups.append(
            {
                "params": image_encoder_params,
                "lr": cfg.scratch.vision_lr,
                "weight_decay": cfg.scratch.weight_decay,
            }
        )

    if optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer

def build_scheduler(optimizer, cfg):
    sch_cfg = cfg.scheduler_schemes[cfg.scratch.scheduler]
    scheduler_type = sch_cfg.type

    if scheduler_type == "cosine":
        eta_min_factor = float(sch_cfg.get("eta_min_factor", 0.1))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sch_cfg.T_max),
            eta_min=cfg.scratch.base_lr * eta_min_factor,
        )

    elif scheduler_type == "step":
        step_size = int(sch_cfg.get("step_size", 10))
        gamma = float(sch_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif scheduler_type == "none":
        scheduler = None

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler

def parse_loss_output(loss_output):
    """
    统一兼容两类损失输出：
    1. dict: 例如 MultiStepMultiMasksAndIous
    2. tensor: 例如 DiceCELoss / DiceCE
    """
    if isinstance(loss_output, dict):
        if "core_loss" in loss_output:
            loss = loss_output["core_loss"]
        else:
            # 兜底：把所有标量项相加
            loss = sum(v for v in loss_output.values() if torch.is_tensor(v))
        loss_dict = loss_output
    elif torch.is_tensor(loss_output):
        loss = loss_output
        loss_dict = {"core_loss": loss_output}
    else:
        raise TypeError(f"Unsupported loss output type: {type(loss_output)}")

    return loss, loss_dict

def compute_loss(loss_fn, outputs, batch, cfg, device):
    loss_name = cfg.scratch.loss_type_name

    if loss_name == "MultiStep":
        loss_output = loss_fn(outputs, batch)
        loss, loss_dict = parse_loss_output(loss_output)
        return loss, loss_dict

    elif loss_name == "DiceCE":
        # 这里先采用“逐帧平均”的方式
        frame_losses = []

        for t, out in enumerate(outputs):
            pred = out["pred_masks_high_res"]                  # [B,1,H,W]
            gt = batch.masks[t].unsqueeze(1).float().to(device)  # [B,1,H,W]
            frame_losses.append(loss_fn(pred, gt))

        loss = sum(frame_losses) / len(frame_losses)
        loss_dict = {"core_loss": loss}
        return loss, loss_dict

    else:
        raise ValueError(f"Unsupported loss_type_name: {loss_name}")

def validate_one_epoch(model, loader, loss_fn, device, cfg):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss, _ = compute_loss(loss_fn, outputs, batch, cfg, device)

            batch_dices = []
            for t, out in enumerate(outputs):
                pred = out["pred_masks_high_res"]  # [B,1,H,W]
                gt = batch.masks[t].unsqueeze(1).float().to(device)  # [B,1,H,W]
                batch_dices.append(compute_batch_dice(pred, gt))

            mean_dice = sum(batch_dices) / len(batch_dices)

            total_loss += loss.item()
            total_dice += mean_dice
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    return avg_loss, avg_dice


def train_one_epoch(model, loader, loss_fn, optimizer, device, cfg):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss, _ = compute_loss(loss_fn, outputs, batch, cfg, device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def main():
    config_dir = r"/home/wusi/SAM2/ESO/CTV/T_20260314/configs"
    config_name = "sam2_ctv_finetune"

    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name=config_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_folds = cfg.scratch.num_folds
    cv_results = []

    for fold_index in range(num_folds):
        fold_dir = Path(cfg.scratch.experiment_log_dir) / f"fold_{fold_index}"
        ckpt_dir = fold_dir / "checkpoints"
        log_dir = fold_dir / "logs"
        tb_dir = fold_dir / "tensorboard"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "fold_log.txt"
        open(log_file, "w", encoding="utf-8").close()

        writer = SummaryWriter(log_dir=str(tb_dir))

        log_message(log_file, "========================================")
        log_message(log_file, f"Starting Fold {fold_index + 1}/{num_folds}")
        log_message(log_file, "========================================")

        # 更新 fold 参数
        cfg.trainer.data.train.fold_index = fold_index
        cfg.trainer.data.train.num_folds = num_folds
        cfg.trainer.data.train.split = "train"

        cfg.trainer.data.val.fold_index = fold_index
        cfg.trainer.data.val.num_folds = num_folds
        cfg.trainer.data.val.split = "val"

        # 实例化 train / val data
        train_data = instantiate(cfg.trainer.data.train, _convert_="all")
        val_data = instantiate(cfg.trainer.data.val, _convert_="all")

        train_loader = train_data.get_loader(epoch=0)
        val_loader = val_data.get_loader(epoch=0)

        # 记录重要训练信息
        train_patient_names = [p.name for p in train_data.dataset.patient_dirs]
        val_patient_names = [p.name for p in val_data.dataset.patient_dirs]

        log_message(log_file, "[Important Config]")
        log_message(log_file, f"train_root_dir: {cfg.scratch.train_root_dir}")
        log_message(log_file, f"experiment_log_dir: {cfg.scratch.experiment_log_dir}")
        log_message(log_file, f"sam2_ckpt_path: {cfg.scratch.sam2_ckpt_path}")
        log_message(log_file, f"freeze_image_encoder: {cfg.scratch.freeze_image_encoder}")
        log_message(log_file, f"clip_len: {cfg.scratch.clip_len}")
        log_message(log_file, f"stride: {cfg.scratch.stride}")
        log_message(log_file, f"num_frames: {cfg.scratch.num_frames}")
        log_message(log_file, f"train_batch_size: {cfg.scratch.train_batch_size}")
        log_message(log_file, f"num_train_workers: {cfg.scratch.num_train_workers}")
        log_message(log_file, f"num_epochs: {cfg.scratch.num_epochs}")
        log_message(log_file, f"base_lr: {cfg.scratch.base_lr}")
        log_message(log_file, f"vision_lr: {cfg.scratch.vision_lr}")
        log_message(log_file, f"weight_decay: {cfg.scratch.weight_decay}")
        log_message(log_file, f"amp_dtype: {cfg.scratch.amp_dtype}")
        log_message(log_file, f"loss_type_name: {cfg.scratch.loss_type_name}")
        log_message(log_file, f"val_every: {cfg.scratch.val_every}")
        log_message(log_file, f"save_last: {cfg.scratch.save_last}")
        log_message(log_file, f"save_best: {cfg.scratch.save_best}")
        log_message(log_file, "")

        log_message(log_file, "[Dataset Split]")
        log_message(log_file, f"num_train_patients: {len(train_patient_names)}")
        log_message(log_file, f"num_val_patients: {len(val_patient_names)}")
        log_message(log_file, f"num_train_clips: {len(train_data.dataset.samples)}")
        log_message(log_file, f"num_val_clips: {len(val_data.dataset.samples)}")
        log_message(log_file, "")

        log_message(log_file, "[Train Patient IDs]")
        for name in train_patient_names:
            log_message(log_file, name)
        log_message(log_file, "")

        log_message(log_file, "[Val Patient IDs]")
        for name in val_patient_names:
            log_message(log_file, name)
        log_message(log_file, "")

        log_message(log_file, "[Train Clips Preview]")
        for i, s in enumerate(train_data.dataset.samples[:30]):
            log_message(
                log_file,
                f"{i}: {s['patient_dir'].name} | z=({s['z_start']},{s['z_end']}) | prompt_mode={s['prompt_mode']}"
            )
        log_message(log_file, "")

        log_message(log_file, "[Val Clips Preview]")
        for i, s in enumerate(val_data.dataset.samples[:30]):
            log_message(
                log_file,
                f"{i}: {s['patient_dir'].name} | z=({s['z_start']},{s['z_end']}) | prompt_mode={s['prompt_mode']}"
            )
        log_message(log_file, "")

        # 实例化 model
        model = instantiate(cfg.trainer.model, _convert_="all")

        # ===== 显式加载 SAM2 预训练权重 =====
        ckpt_path = cfg.scratch.sam2_ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        model = model.to(device)

        log_message(log_file, f"[Checkpoint] Loaded from: {ckpt_path}")
        log_message(log_file, f"[Checkpoint] Missing keys: {len(missing)}")
        for k in missing[:30]:
            log_message(log_file, f"  missing: {k}")

        log_message(log_file, f"[Checkpoint] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:30]:
            log_message(log_file, f"  unexpected: {k}")

        # optimizer
        optimizer = build_optimizer(model, cfg)

        # scheduler
        scheduler = build_scheduler(optimizer, cfg)

        # loss
        loss_fn = instantiate(cfg.trainer.loss["all"], _convert_="all")

        best_metric = -1.0
        best_epoch = -1

        for epoch in range(cfg.scratch.num_epochs):
            log_message(log_file, f"\n=== Fold {fold_index} | Epoch {epoch + 1}/{cfg.scratch.num_epochs} ===")

            train_loader = train_data.get_loader(epoch)
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
            )

            writer.add_scalar("train/loss", train_loss, epoch)
            log_message(log_file, f"[Fold {fold_index}] Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

            if (epoch + 1) % cfg.scratch.val_every == 0:
                val_loader = val_data.get_loader(epoch)
                val_loss, val_dice = validate_one_epoch(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    cfg=cfg,
                )

                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/dice", val_dice, epoch)

                log_message(
                    log_file,
                    f"[Fold {fold_index}] Epoch {epoch + 1} Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
                )

                if cfg.scratch.save_best and val_dice > best_metric:
                    best_metric = val_dice
                    best_epoch = epoch
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        save_path=ckpt_dir / "best.pth",
                    )
                    log_message(
                        log_file,
                        f"[Fold {fold_index}] New best model saved. Dice = {best_metric:.4f} at epoch {epoch + 1}"
                    )

            if cfg.scratch.save_last:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    save_path=ckpt_dir / "last.pth",
                )

            if scheduler is not None:
                scheduler.step()
                current_lrs = [group["lr"] for group in optimizer.param_groups]
                log_message(log_file, f"[Fold {fold_index}] Epoch {epoch + 1} LR: {current_lrs}")

        writer.close()
        log_message(log_file, f"\nFold {fold_index} finished. Best Dice = {best_metric:.4f}\n")

        cv_results.append(
            {
                "fold": fold_index,
                "best_dice": best_metric,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else -1,
                "best_ckpt": str(ckpt_dir / "best.pth"),
                "last_ckpt": str(ckpt_dir / "last.pth"),
            }
        )

    # ===== 保存五折汇总结果 =====
    summary_path = Path(cfg.scratch.cv_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "best_dice", "best_epoch", "best_ckpt", "last_ckpt"]
        )
        writer.writeheader()
        writer.writerows(cv_results)

    # 选出 best fold
    valid_results = [x for x in cv_results if x["best_epoch"] != -1]
    if len(valid_results) > 0:
        best_fold_result = max(valid_results, key=lambda x: x["best_dice"])

        best_txt_path = summary_path.parent / "best_fold.txt"
        with open(best_txt_path, "w", encoding="utf-8") as f:
            f.write(f"best_fold: {best_fold_result['fold']}\n")
            f.write(f"best_dice: {best_fold_result['best_dice']:.6f}\n")
            f.write(f"best_epoch: {best_fold_result['best_epoch']}\n")
            f.write(f"best_ckpt: {best_fold_result['best_ckpt']}\n")

        print("\n========================================")
        print("Cross-validation summary saved.")
        print(f"Best fold: {best_fold_result['fold']}")
        print(f"Best dice: {best_fold_result['best_dice']:.6f}")
        print(f"Best checkpoint: {best_fold_result['best_ckpt']}")
        print("========================================\n")
    else:
        print("\nNo valid best fold found.\n")

if __name__ == "__main__":
    main()