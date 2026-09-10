#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_core import stratified_spaced_prompt_indices
from io_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT_PATH,
    DEFAULT_TARGET_Z_SPACING,
    RectalCTVVolumeDataset,
    append_table_txt,
    append_txt_log,
    build_model,
    build_optimizer,
    build_scheduler,
    ctv_case_name,
    list_patient_dirs,
    load_checkpoint,
    load_preprocessed_image_ctv,
    make_or_load_splits,
    patient_id_from_dir,
    plot_loss_curve,
    save_checkpoint,
    set_global_seed,
)
from loops import train_one_epoch, validate_fixed_plan
from training.utils.data_utils import collate_fn


def collate_one(batch):
    return collate_fn(batch, dict_key="eso_ctv_k1_5")


def validate_preprocess_manifest(args) -> None:
    path = args.data_root / "preprocess_manifest_train.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Offline preprocessing manifest not found: {path}. Run preprocess.py first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "partition": "train",
        "target_z_spacing": float(args.target_z_spacing),
        "window_center": float(args.window_center),
        "window_width": float(args.window_width),
        "normalized_range": [0.0, 1.0],
        "image_size": int(args.input_size),
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError(f"Offline preprocessing manifest mismatch: {mismatched}")


def _mask_frame_info(pdir: Path) -> tuple[int, list[int]]:
    _, mask = load_preprocessed_image_ctv(pdir, expected_size=512)
    positive = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1)).tolist()
    if not positive:
        raise ValueError(f"CTV mask has no positive slice: {pdir / 'CTV.nii.gz'}")
    return int(mask.shape[0]), [int(x) for x in positive]


def make_or_load_validation_plan(
    path: Path,
    splits: list[dict],
    seed: int,
    min_prompt_gap: int,
    target_z_spacing: float,
) -> dict:
    """Persist two placements for every fold/validation patient/K forever."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) != 4:
            raise ValueError(
                f"Validation plan {path} is from an incompatible sampler. "
                "Move it aside and rerun to generate the SI-stratified positive-slice plan."
            )
        if int(data["min_prompt_gap"]) != int(min_prompt_gap):
            raise ValueError(
                f"Saved plan uses min_prompt_gap={data['min_prompt_gap']}, requested {min_prompt_gap}"
            )
        if float(data.get("target_z_spacing", -1.0)) != float(target_z_spacing):
            raise ValueError("Saved validation plan uses a different target Z spacing")
        for fold_info in splits:
            fold_key = str(int(fold_info["fold"]))
            if fold_key not in data.get("folds", {}):
                raise ValueError(f"Validation plan is missing fold {fold_key}")
            saved_fold = data["folds"][fold_key]
            for pdir in fold_info["val"]:
                patient_id = patient_id_from_dir(pdir)
                patient_key = str(patient_id)
                if patient_key not in saved_fold:
                    raise ValueError(
                        f"Validation plan is missing {ctv_case_name(pdir)} in fold {fold_key}"
                    )
                num_frames, positive_indices = _mask_frame_info(pdir)
                positive_set = set(positive_indices)
                record = saved_fold[patient_key]
                if int(record.get("num_frames", -1)) != num_frames:
                    raise ValueError(f"Frame count changed for {ctv_case_name(pdir)}")
                if record.get("ctv_positive_frame_ids") != positive_indices:
                    raise ValueError(f"CTV-positive slices changed for {ctv_case_name(pdir)}")
                for k in range(1, 6):
                    placements = record.get("placements", {}).get(str(k), [])
                    if len(placements) != 2:
                        raise ValueError(
                            f"{ctv_case_name(pdir)}, K={k} must have exactly two placements"
                        )
                    seen = set()
                    for placement in placements:
                        prompts = [int(x) for x in placement.get("prompt_frame_ids", [])]
                        if len(prompts) != k:
                            raise ValueError(
                                f"{ctv_case_name(pdir)}, K={k} has {len(prompts)} prompts"
                            )
                        if not set(prompts).issubset(positive_set):
                            raise ValueError(
                                f"{ctv_case_name(pdir)}, K={k} contains CTV-negative prompts"
                            )
                        if any(
                            b - a < int(min_prompt_gap)
                            for a, b in zip(prompts[:-1], prompts[1:])
                        ):
                            raise ValueError(
                                f"{ctv_case_name(pdir)}, K={k} violates min prompt gap"
                            )
                        seen.add(tuple(prompts))
                    if len(seen) != 2:
                        raise ValueError(
                            f"{ctv_case_name(pdir)}, K={k} placements are not distinct"
                        )
        return data

    folds_data = {}
    for fold_info in splits:
        fold = int(fold_info["fold"])
        patients = {}
        for pdir in fold_info["val"]:
            patient_id = patient_id_from_dir(pdir)
            num_frames, positive_indices = _mask_frame_info(pdir)
            by_k = {}
            for k in range(1, 6):
                placements = []
                seen = set()
                attempt = 0
                while len(placements) < 2 and attempt < 10000:
                    placement_seed = (
                        int(seed)
                        + fold * 10000019
                        + patient_id * 10007
                        + k * 101
                        + attempt
                    )
                    indices = stratified_spaced_prompt_indices(
                        positive_indices,
                        k,
                        min_prompt_gap,
                        random.Random(placement_seed),
                    )
                    key = tuple(indices)
                    attempt += 1
                    if key in seen:
                        continue
                    seen.add(key)
                    placements.append(
                        {
                            "placement_id": len(placements) + 1,
                            "seed": placement_seed,
                            "prompt_frame_ids": indices,
                        }
                    )
                if len(placements) != 2:
                    raise ValueError(
                        f"Patient {ctv_case_name(pdir)}, K={k} cannot produce two distinct "
                        f"SI-stratified placements from positive slices {positive_indices} "
                        f"with min_prompt_gap={min_prompt_gap}."
                    )
                by_k[str(k)] = placements
            patients[str(patient_id)] = {
                "patient": ctv_case_name(pdir),
                "num_frames": num_frames,
                "ctv_positive_frame_ids": positive_indices,
                "placements": by_k,
            }
        folds_data[str(fold)] = patients

    data = {
        "schema_version": 4,
        "seed": int(seed),
        "num_placements_per_patient_k": 2,
        "prompt_ks": [1, 2, 3, 4, 5],
        "min_prompt_gap": int(min_prompt_gap),
        "target_z_spacing": float(target_z_spacing),
        "folds": folds_data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def run_one_fold(args, fold_info: dict, full_plan: dict, device: torch.device):
    fold = int(fold_info["fold"])
    run_dir = args.output_root / "Mixed_K1_5" / f"fold_{fold}"
    ckpt_dir = run_dir / "checkpoints"
    log_txt = run_dir / "train_log.txt"
    metrics_txt = run_dir / "metrics_log.txt"
    val_txt = run_dir / "val_by_k.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_flag = run_dir / "completed.flag"
    if completed_flag.exists():
        msg = f"[Mixed_K1_5 fold {fold}] completed.flag found; skipping completed fold."
        print(msg, flush=True)
        append_txt_log(log_txt, msg)
        return

    set_global_seed(args.seed + fold)
    model, trainable_stats = build_model(args.model_cfg, args.init_ckpt, device, args)
    if not bool(getattr(model, "pred_obj_scores", False)):
        raise RuntimeError(
            "This training design requires SAM2 pred_obj_scores=True so that the "
            "presence BCE and native no-object memory pathway remain active."
        )
    (run_dir / "trainable_stats.json").write_text(
        json.dumps(trainable_stats, indent=2), encoding="utf-8"
    )
    train_ds = RectalCTVVolumeDataset(
        fold_info["train"], input_size=args.input_size, window_center=args.window_center,
        window_width=args.window_width
    )
    val_ds = RectalCTVVolumeDataset(
        fold_info["val"], input_size=args.input_size, window_center=args.window_center,
        window_width=args.window_width
    )
    generator = torch.Generator().manual_seed(args.seed + fold)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_one, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_one
    )
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.max_epochs, args.warmup_epochs, args.min_lr)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.amp and args.amp_dtype == torch.float16)
    )

    start_epoch, best_metric, best_epoch, patience_counter = 0, -1.0, -1, 0
    latest_path = ckpt_dir / "latest.pth"
    if args.resume and latest_path.exists():
        start_epoch, best_metric, best_epoch, patience_counter = load_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device=device,
            return_patience=True,
        )
        print(
            f"[Mixed_K1_5 fold {fold}] resume from epoch={start_epoch}, "
            f"best={best_metric:.4f}, best_epoch={best_epoch}, "
            f"patience={patience_counter}/{args.patience}",
            flush=True,
        )
    validation_plan = {
        patient_id: record["placements"]
        for patient_id, record in full_plan["folds"][str(fold)].items()
    }
    # Track the last fully completed epoch robustly, including resume-at-max-epoch.
    last_epoch = int(start_epoch)

    # If a previous run already reached the early-stopping condition but was
    # interrupted before completed.flag was written, mark the fold complete now.
    if patience_counter >= args.patience:
        completed_flag.write_text(
            json.dumps(
                {
                    "fold": fold,
                    "last_epoch": last_epoch,
                    "best_metric": float(best_metric),
                    "best_epoch": int(best_epoch),
                    "patience_counter": int(patience_counter),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        msg = (
            f"[Mixed_K1_5 fold {fold}] resume state already satisfies early stopping; "
            f"marking fold complete."
        )
        print(msg, flush=True)
        append_txt_log(log_txt, msg)
        return

    for epoch in range(start_epoch, args.max_epochs):
        generator.manual_seed(args.seed + fold * 100 + epoch)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args.amp_dtype,
            amp_enabled=args.amp,
            min_prompts=1, max_prompts=5, min_prompt_gap=args.min_prompt_gap,
            use_bidirectional_train=args.bidirectional_train,
            grad_clip_norm=args.grad_clip_norm,
            forward_backbone_per_frame=args.forward_backbone_per_frame,
            epoch=epoch, seed=args.seed + fold * 10000,
            presence_loss_weight=args.presence_loss_weight,
        )
        epoch_num = epoch + 1
        last_epoch = int(epoch_num)
        should_validate = (
            epoch_num == 1
            or epoch_num % int(args.val_interval) == 0
            or epoch_num == int(args.max_epochs)
        )

        val_results = None
        val_metric = float("nan")

        if should_validate:
            val_results, val_metric = validate_fixed_plan(
                model, val_loader, device, args.amp_dtype, validation_plan,
                prompt_ks=range(1, 6),
                forward_backbone_per_frame=args.forward_backbone_per_frame,
                amp_enabled=args.amp,
            )

        scheduler.step()

        if should_validate:
            improved = val_metric > best_metric
            if improved:
                best_metric, best_epoch, patience_counter = val_metric, epoch_num, 0
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    epoch_num,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_metric,
                    best_epoch,
                    vars(args),
                    patience_counter=patience_counter,
                )
            else:
                patience_counter += 1

        save_checkpoint(
            latest_path,
            epoch_num,
            model,
            optimizer,
            scheduler,
            scaler,
            best_metric,
            best_epoch,
            vars(args),
            patience_counter=patience_counter,
        )

        append_table_txt(metrics_txt, {
            "fold": fold, "epoch": epoch_num, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_seg_loss": train_stats["seg_loss"],
            "train_presence_loss": train_stats["presence_loss"],
            "train_unprompted_slice_3d_dsc": train_stats["unprompted_slice_3d_dsc"],
            "val_unprompted_slice_3d_dsc_mean": val_metric,
            "best_metric": best_metric, "best_epoch": best_epoch,
        })

        if should_validate:
            for result in val_results:
                append_table_txt(val_txt, {
                    "fold": fold, "epoch": epoch_num, "k": result.k,
                    "placements": 2,
                    "unprompted_slice_3d_dsc": result.unprompted_slice_3d_dsc,
                    "whole_volume_3d_dsc": result.whole_volume_3d_dsc,
                    "unprompted_mean_2d_dice_aux": result.unprompted_mean_2d_dice,
                })

        plot_loss_curve(metrics_txt, run_dir / "loss_curve.png")

        if should_validate:
            msg = (
                f"[Mixed_K1_5 fold {fold}] epoch {epoch_num}/{args.max_epochs} "
                f"loss={train_stats['loss']:.4f} val={val_metric:.4f} "
                f"best={best_metric:.4f} best_epoch={best_epoch} "
                f"patience={patience_counter}/{args.patience}"
            )
        else:
            msg = (
                f"[Mixed_K1_5 fold {fold}] epoch {epoch_num}/{args.max_epochs} "
                f"loss={train_stats['loss']:.4f} val=SKIP "
                f"best={best_metric:.4f} best_epoch={best_epoch}"
            )

        print(msg)
        append_txt_log(log_txt, msg)

        if should_validate and patience_counter >= args.patience:
            append_txt_log(
                log_txt,
                f"[EARLY STOP] epoch {epoch_num}: "
                f"{patience_counter} consecutive validation rounds without improvement"
            )
            break

    completed_flag.write_text(
        json.dumps(
            {
                "fold": fold,
                "last_epoch": int(last_epoch),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "patience_counter": int(patience_counter),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    done_msg = (
        f"[Mixed_K1_5 fold {fold}] COMPLETE "
        f"best={best_metric:.4f} best_epoch={best_epoch}"
    )
    print(done_msg, flush=True)
    append_txt_log(log_txt, done_msg)


def main():
    parser = argparse.ArgumentParser("Train ESO CTV SAM2-LoRA with dynamic K=1..5 prompts")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--val-interval",
        type=int,
        default=10,
        help="Run full validation at epoch 1, every N epochs, and the final epoch.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early-stopping patience measured in validation rounds, not epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument("--target-z-spacing", type=float, default=DEFAULT_TARGET_Z_SPACING)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--min-prompt-gap", type=int, default=2)
    parser.add_argument("--presence-loss-weight", type=float, default=0.05)
    parser.add_argument("--bidirectional-train", action="store_true")
    parser.add_argument(
        "--forward-backbone-per-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use per-frame Image Encoder forward during training to reduce peak memory. "
            "Enabled by default; disable explicitly with --no-forward-backbone-per-frame."
        ),
    )
    parser.add_argument(
        "--image-encoder-activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    if args.input_size != 512:
        raise ValueError("Offline Stage1 preprocessing is fixed at input_size=512")
    if args.presence_loss_weight < 0:
        raise ValueError("presence_loss_weight must be non-negative")
    if args.val_interval <= 0:
        raise ValueError("val_interval must be a positive integer")
    if args.patience <= 0:
        raise ValueError("patience must be a positive integer")
    args.amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)
    validate_preprocess_manifest(args)
    patient_dirs = list_patient_dirs(args.data_root / "train")
    splits = make_or_load_splits(
        patient_dirs, args.num_folds, args.seed, args.split_path
    )
    plan = make_or_load_validation_plan(
        args.output_root / "validation_prompt_plan.json",
        splits, args.seed, args.min_prompt_gap, args.target_z_spacing
    )
    for fold_info in splits:
        run_one_fold(args, fold_info, plan, device)


if __name__ == "__main__":
    main()
