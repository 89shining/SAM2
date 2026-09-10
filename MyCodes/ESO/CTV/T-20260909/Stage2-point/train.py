#!/usr/bin/env python
"""SAM2 Stage-2 point-correction training. Importing never starts a run."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from stage1_bridge import (
    DEFAULT_DATA_ROOT, DEFAULT_INIT_CKPT, DEFAULT_MODEL_CFG, DEFAULT_SPLIT_PATH,
    RectalCTVVolumeDataset, build_model, list_patient_dirs,
    make_or_load_splits, set_global_seed,
)
from stage2_loops import train_patient_episode, validate_k3_t5
from training.utils.data_utils import collate_fn


STAGE1_RESULTS = Path("/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage1-mask/TrainResults")
DEFAULT_OUTPUT_ROOT = Path("/home/intern/ftp/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage2-point/TrainResults")


def collate_one(items):
    return collate_fn(items, dict_key="eso_ctv_stage2_point")


def _patient_id(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[1])


def _spacing_map(patient_dirs) -> dict[int, tuple[float, float, float]]:
    result = {}
    for patient_dir in patient_dirs:
        sx, sy, sz = sitk.ReadImage(str(patient_dir / "image.nii.gz")).GetSpacing()
        result[_patient_id(patient_dir)] = (float(sz), float(sy), float(sx))
    return result


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_stage1_state(model, checkpoint_path: Path) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage1 best checkpoint not found: {checkpoint_path}")
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Not a Stage1 checkpoint containing model weights: {checkpoint_path}")
    incompatible = model.load_state_dict(state["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Stage1 checkpoint mismatch: missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return state


def _make_model(args, device):
    model, stats = build_model(args.model_cfg, args.init_ckpt, device, args)
    stage1_state = _load_stage1_state(model, args.stage1_ckpt)
    checkpoint_args = stage1_state.get("args", {})
    checkpoint_fold = checkpoint_args.get("fold") if isinstance(checkpoint_args, dict) else None
    if checkpoint_fold is not None and int(checkpoint_fold) != int(args.fold):
        raise RuntimeError(
            f"Requested Stage2 fold {args.fold}, but Stage1 checkpoint records fold {checkpoint_fold}"
        )
    if not bool(getattr(model, "pred_obj_scores", False)):
        raise RuntimeError("Stage2 must retain Stage1 pred_obj_scores=True")
    return model, stats


def _save_checkpoint(path, epoch, model, optimizer, scheduler, best, best_epoch, patience, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "best_metric": float(best), "best_epoch": int(best_epoch),
        "patience_counter": int(patience), "args": vars(args),
        "stage1_checkpoint": str(args.stage1_ckpt),
    }, str(path))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--stage1-ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT / "train")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--validation-plan", type=Path, default=STAGE1_RESULTS / "validation_prompt_plan.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--image-encoder-activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.fold not in range(5):
        raise ValueError(f"--fold must be one of 0,1,2,3,4; got {args.fold}")
    if args.input_size != 512:
        raise ValueError("Stage2 must read the same 512x512 offline preprocessing as Stage1")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)
    all_patients = list_patient_dirs(args.data_root)
    splits = make_or_load_splits(all_patients, 5, args.seed, args.split_path)
    matches = [item for item in splits if int(item["fold"]) == args.fold]
    if len(matches) != 1:
        raise ValueError(f"Fold {args.fold} is missing from {args.split_path}")
    fold_info = matches[0]
    full_plan = json.loads(args.validation_plan.read_text(encoding="utf-8"))
    fold_plan = full_plan.get("folds", {}).get(str(args.fold))
    if fold_plan is None:
        raise KeyError(f"Fold {args.fold} is missing from the Stage1 validation plan")
    expected_val_ids = {_patient_id(Path(path)) for path in fold_info["val"]}
    plan_val_ids = {int(patient_id) for patient_id in fold_plan}
    if plan_val_ids != expected_val_ids:
        raise RuntimeError(
            "Stage1 validation-plan patients do not match the selected split: "
            f"missing={sorted(expected_val_ids - plan_val_ids)}, "
            f"unexpected={sorted(plan_val_ids - expected_val_ids)}"
        )

    run_dir = args.output_root / "Stage2_T0_5" / f"fold_{args.fold}"
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "completed.flag"
    if completed.exists():
        print(f"[Stage2 fold {args.fold}] completed.flag found; skipping", flush=True); return
    (run_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    model, stats = _make_model(args, device)
    (run_dir / "trainable_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    train_ds = RectalCTVVolumeDataset(fold_info["train"], input_size=512)
    val_ds = RectalCTVVolumeDataset(fold_info["val"], input_size=512)
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_one, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_one)
    spacing_by_patient = _spacing_map(all_patients)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    latest, best_path = ckpt_dir / "latest.pth", ckpt_dir / "best.pth"
    start, best, best_epoch, patience = 1, float("-inf"), 0, 0
    if args.resume and latest.exists():
        state = torch.load(str(latest), map_location=device, weights_only=False)
        if str(state.get("stage1_checkpoint")) != str(args.stage1_ckpt):
            raise RuntimeError("Resume checkpoint was initialized from a different Stage1 checkpoint")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start, best, best_epoch, patience = int(state["epoch"]) + 1, float(state["best_metric"]), int(state["best_epoch"]), int(state["patience_counter"])

    print({"fold": args.fold, "stage1_ckpt": str(args.stage1_ckpt), "train_T": "U(0..5)", "validation": "K=3 x two placements, D0..D5; best=mean(D0..D5)", "lr": args.lr}, flush=True)
    for epoch in range(start, args.epochs + 1):
        model.train(); losses = []; loss_seg_values = []; loss_presence_values = []; terminal_dice = []
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            patient = int(batch.metadata.unique_objects_identifier[0, 0, 0].item())
            rng = random.Random(args.seed + args.fold * 10000019 + epoch * 1000003 + patient)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, trace = train_patient_episode(model, batch, spacing_by_patient[patient], rng)
            for click in trace["clicks"]:
                _append_csv(run_dir / "correction_click_log.csv", {
                    "patient_id": patient, "epoch": epoch,
                    "K": len(trace["initial_prompt_frames"]),
                    "initial_prompt_slices": trace["initial_prompt_frames"],
                    "sampled_T": trace["sampled_T"], **click,
                })
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for patient {patient}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            loss_seg_values.append(float(trace["loss_seg"]))
            loss_presence_values.append(float(trace["loss_presence"]))
            terminal_dice.append(float(trace["terminal_dice"]))
        scheduler.step()

        should_validate = epoch == 1 or epoch % 10 == 0 or epoch == args.epochs
        val_dice = None
        if should_validate:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                val_dice, rows, val_summary = validate_k3_t5(
                    model, val_loader, fold_plan, spacing_by_patient
                )
            for row in rows:
                _append_csv(run_dir / "validation_patient_level.csv", {"epoch": epoch, **row})
            _append_csv(run_dir / "validation_summary.csv", {"epoch": epoch, **val_summary})
            if val_dice > best:
                best, best_epoch, patience = val_dice, epoch, 0
                _save_checkpoint(best_path, epoch, model, optimizer, scheduler, best, best_epoch, patience, args)
            else:
                patience += 1
        _save_checkpoint(latest, epoch, model, optimizer, scheduler, best, best_epoch, patience, args)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_loss_seg = float(np.mean(loss_seg_values)) if loss_seg_values else float("nan")
        train_loss_presence = float(np.mean(loss_presence_values)) if loss_presence_values else float("nan")
        train_dice = float(np.mean(terminal_dice)) if terminal_dice else float("nan")
        _append_csv(run_dir / "training_log.csv", {"epoch": epoch, "loss_total": train_loss, "loss_seg": train_loss_seg, "loss_presence": train_loss_presence, "terminal_dice": train_dice, "workflow_score_p0_p5": "" if val_dice is None else val_dice, "best": best, "best_epoch": best_epoch, "patience": patience})
        print(f"[Stage2 fold {args.fold}] epoch {epoch}/{args.epochs} loss={train_loss:.5f} terminal_dice={train_dice:.5f} val={val_dice} best={best:.5f}@{best_epoch}", flush=True)
        if should_validate and patience >= args.patience:
            break
    completed.write_text(json.dumps({"fold": args.fold, "best_metric": best, "best_epoch": best_epoch}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
