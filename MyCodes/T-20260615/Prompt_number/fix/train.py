#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from io_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    RectalCTVVolumeDataset,
    append_txt_log,
    append_table_txt,
    build_model,
    build_optimizer,
    build_scheduler,
    ctv_case_name,
    default_result_base,
    list_patient_dirs,
    load_checkpoint,
    make_or_load_splits,
    plot_loss_curve,
    save_checkpoint,
    set_global_seed,
    write_prompt_record_excel,
)
from loops import train_one_epoch, validate_fixed_ks
from training.utils.data_utils import collate_fn


def collate_one(batch):
    return collate_fn(batch, dict_key="rectal_ctv")


def parse_models(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x.strip()) for x in part.split("-", 1)]
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def run_one_fold(args, model_x: int, fold_info: dict, device: torch.device):
    fold = int(fold_info["fold"])
    run_dir = args.output_root / f"Model_{model_x}" / f"fold_{fold}"
    ckpt_dir = run_dir / "checkpoints"
    log_txt = run_dir / "train_log.txt"
    metrics_txt = run_dir / "metrics_log.txt"
    val_txt = run_dir / "val_by_k.txt"
    curve_png = run_dir / "loss_curve.png"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed + model_x * 1000 + fold)
    model, trainable_stats = build_model(args.model_cfg, args.init_ckpt, device, args)
    (run_dir / "trainable_stats.json").write_text(json.dumps(trainable_stats, indent=2), encoding="utf-8")
    append_txt_log(log_txt, "=" * 80)
    append_txt_log(log_txt, f"Start Model_{model_x} fold {fold}")
    append_txt_log(log_txt, f"train patients: {[ctv_case_name(p) for p in fold_info['train']]}")
    append_txt_log(log_txt, f"val patients: {[ctv_case_name(p) for p in fold_info['val']]}")
    append_txt_log(log_txt, f"trainable stats: {trainable_stats}")

    train_ds = RectalCTVVolumeDataset(
        fold_info["train"],
        input_size=args.input_size,
        window_center=args.window_center,
        window_width=args.window_width,
    )
    val_ds = RectalCTVVolumeDataset(
        fold_info["val"],
        input_size=args.input_size,
        window_center=args.window_center,
        window_width=args.window_width,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + model_x * 1000 + fold)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_one,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_one,
    )

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.max_epochs, args.warmup_epochs, args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    start_epoch = 0
    best_metric = -1.0
    best_epoch = -1
    latest_path = ckpt_dir / "latest.pth"
    if args.resume and latest_path.exists():
        start_epoch, best_metric, best_epoch = load_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, device=device
        )
        msg = f"[RESUME] Model_{model_x} fold {fold}: epoch={start_epoch}, best={best_metric:.4f}"
        print(msg)
        append_txt_log(log_txt, msg)

    patience_counter = 0
    for epoch in range(start_epoch, args.max_epochs):
        if hasattr(train_loader.sampler, "generator") and train_loader.sampler.generator is not None:
            train_loader.sampler.generator.manual_seed(args.seed + model_x * 1000 + fold * 100 + epoch)
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=args.amp_dtype,
            max_prompts=model_x,
            use_bidirectional_train=args.bidirectional_train,
            grad_clip_norm=args.grad_clip_norm,
            forward_backbone_per_frame=args.forward_backbone_per_frame,
            epoch=epoch,
            seed=args.seed + fold * 10000,
        )

        val_results, val_metric = validate_fixed_ks(
            model=model,
            loader=val_loader,
            device=device,
            amp_dtype=args.amp_dtype,
            max_prompts=model_x,
            forward_backbone_per_frame=args.forward_backbone_per_frame,
        )
        scheduler.step()

        improved = val_metric > best_metric
        if improved:
            best_metric = val_metric
            best_epoch = epoch + 1
            patience_counter = 0
            save_checkpoint(
                ckpt_dir / "best.pth",
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                best_metric,
                best_epoch,
                vars(args),
            )
        else:
            patience_counter += 1

        save_checkpoint(
            latest_path,
            epoch + 1,
            model,
            optimizer,
            scheduler,
            scaler,
            best_metric,
            best_epoch,
            vars(args),
        )

        row = {
            "model_x": model_x,
            "fold": fold,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_unprompted_slice_3d_dsc": train_stats["unprompted_slice_3d_dsc"],
            "val_unprompted_slice_3d_dsc_mean": val_metric,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        append_table_txt(metrics_txt, row)
        for result in val_results:
            append_table_txt(
                val_txt,
                {
                    "model_x": model_x,
                    "fold": fold,
                    "epoch": epoch + 1,
                    "k": result.k,
                    "unprompted_slice_3d_dsc": result.unprompted_slice_3d_dsc,
                    "whole_volume_3d_dsc": result.whole_volume_3d_dsc,
                    "unprompted_mean_2d_dice_aux": result.unprompted_mean_2d_dice,
                },
            )
        plot_loss_curve(metrics_txt, curve_png)
        msg = (
            f"[Model_{model_x} fold {fold}] epoch {epoch + 1}/{args.max_epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"loss={train_stats['loss']:.4f} "
            f"train_unprompted_3d={train_stats['unprompted_slice_3d_dsc']:.4f} "
            f"val_unprompted_3d={val_metric:.4f} "
            f"best={best_metric:.4f} best_epoch={best_epoch}"
        )
        print(msg)
        append_txt_log(log_txt, msg)

        if patience_counter >= args.patience:
            msg = f"[EARLY STOP] Model_{model_x} fold {fold} at epoch {epoch + 1}"
            print(msg)
            append_txt_log(log_txt, msg)
            break


def main():
    parser = argparse.ArgumentParser("Train SAM2-LoRA prompt-number experiment")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--models", type=str, default="2-6")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional-train", action="store_true")
    parser.add_argument("--forward-backbone-per-frame", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = default_result_base(args.data_root) / "TrainResults"
    args.output_root = args.output_root.resolve()
    args.amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_global_seed(args.seed)

    patient_dirs = list_patient_dirs(args.data_root / "train_nii")
    splits = make_or_load_splits(
        patient_dirs,
        args.num_folds,
        args.seed,
        args.output_root / "splits.json",
    )
    model_xs = parse_models(args.models)
    prompt_record_path = args.output_root / "train_prompt_frames.xlsx"
    if not prompt_record_path.exists():
        print(f"[INFO] writing train prompt frames: {prompt_record_path}")
        write_prompt_record_excel(
            prompt_record_path,
            patient_dirs,
            range(2, max(model_xs) + 1),
            "train",
        )

    for model_x in model_xs:
        for fold_info in splits:
            run_one_fold(args, model_x, fold_info, device)


if __name__ == "__main__":
    main()
