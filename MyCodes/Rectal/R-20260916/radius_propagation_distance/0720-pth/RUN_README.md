# R-20260916: POS/NEG physical disk and physical-gate ablation

This directory contains code only. Training and inference must be started
explicitly; no script is running merely because the directory exists.

## Locked inputs

- Source data: `/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask`
- Split: `R-20260720/TrainResults/POS/split.json` (MD5
  `2933e8e1ef887ecc62606ffa3c5a3eb0`)
- POS initialization: `R-20260720/TrainResults/POS/checkpoints/best.pth`
- NEG initialization: `R-20260720/TrainResults/NEG/checkpoints/best.pth`
- Input: 512; temporal training window: 9; maximum epochs: 60; LR warmup: 5;
  patience: 10 starts accumulating only after epoch 5.

## 1. Build isolated derived prompts

```bash
PY=/home/wusi/miniconda3/envs/sam2/bin/python
CODE=/home/wusi/SAM2/MyTrain/MyCodes/Rectal/R-20260916
OUT=/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260916
SRC=/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask
SPLIT=/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/TrainResults/POS/split.json

$PY "$CODE/generate_physical_disk_prompts.py" \
  --source-root "$SRC" --output-root "$OUT" --split-json "$SPLIT" \
  --radii-mm 2 4 6 8 10 12 14 16 18 20
touch "$OUT/PROMPTS_READY"

# Fail-closed preflight. It creates PREFLIGHT_PASS only when all checks pass.
$PY "$CODE/preflight_audit.py" \
  --data-root "$OUT/Prompt_mask" --split-json "$SPLIT" \
  --output-root "$OUT" --code-root "$CODE"

# Deterministic 50% layers must be identical over all r for each patient/branch.
$PY "$CODE/prompt_slice_consistency_audit.py" \
  --data-root "$OUT/Prompt_mask" --split-json "$SPLIT" \
  --output "$OUT/metrics/prompt_slice_consistency_audit.csv"
```

The derived `Prompt_mask` contains symlinks to immutable base images plus new
postprocessed masks and disks. A disk has no component/GT/nnU-Net/body clipping.

## 2. Train the 20 independent branch models

```bash
cd "$OUT"
SAM2_PROJECT_ROOT=/home/wusi/SAM2 \
  "$PY" "$CODE/schedule_training.py" >> logs/training_scheduler.log 2>&1
```

The scheduler uses at most one task per GPU and up to 8 concurrent tasks. It
requires at least 16 GiB free GPU memory and utilization at most 20%; it will
therefore use all eligible idle A100 GPUs without touching a busy GPU. Each
model's best checkpoint is selected only by the fixed validation training Dice.

## 3. Cache fixed-checkpoint validation probabilities

After all `checkpoints/{pos,neg}/rXXmm/DONE` markers exist:

```bash
SAM2_PROJECT_ROOT=/home/wusi/SAM2 \
  "$PY" "$CODE/run_validation_inference.py" >> "$OUT/logs/validation_inference.log" 2>&1
```

This runs deterministic 50% prompt inference at the model's matched disk
radius. It does not apply a propagation gate.

## 4. Independently select POS and NEG configurations

```bash
for BRANCH in pos neg; do
  "$PY" "$CODE/select_branch_validation.py" \
    --kind "$BRANCH" --data-root "$OUT/Prompt_mask" --split-json "$SPLIT" \
    --cache-root "$OUT/probability_cache/validation" --metrics-root "$OUT/metrics"
done
```

For every fixed checkpoint, it evaluates
`m={0,5,10,15,20,30,40,60,infinity} mm`, first selects `m*(r)`, then selects
`r*`. Selection is Dice within 0.001 of the maximum, then lower mean HD95, then
smaller spatial parameter. `m=0` is asserted to equal Disk-only voxel-for-voxel;
infinity is an explicit all-true gate. POS and NEG remain separate in this stage.

After the validation profile is reviewed, each prelocked `(r,m*(r))` may be
evaluated on test to form a radius sensitivity curve. The validation-locked `r*`
remains the sole formal configuration even if another point is higher on test.
Only each radius's prelocked best-gate NIfTI predictions are retained. No POS+NEG
fusion is launched by these commands.
