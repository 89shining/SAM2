# R-20260922 Skeleton-thickness sensitivity analysis

## Locked method

Source masks are already-created 2-D axial skeletons. For each selected prompt slice, skeleton points are decomposed into 8-neighbour components; each component contributes a contiguous weighted-geodesic segment. Training selects a random 30–70% segment; validation/test use the central 50%. The segment is physically thickened in 2-D with true XY spacing by `distance <= thickness`, without clipping to raw errors, GT, nnU-Net, or component boundaries. Thicknesses are `0,2,4,6,8,10,12 mm` (0 is the binary skeleton).

Axial-slice selection is independent: training random patient-level 30–70%; validation fixed deterministic uniform 50%; test deterministic uniform 0/25/50/75/100%. POS and NEG train independently. Training uses 512 input, a 9-slice window, and one random propagation direction; validation/test use forward/backward probability means. No propagation call receives GT masks.

Each thickness trains one POS and one NEG model from the original `sam2.1_hiera_small.pt` (14 models). `m={0,5,10,15,20,30,40,60,infinity} mm` is an offline 3-D physical gate, not an additional training dimension. Selection is strictly `best epoch -> best m for each thickness -> best thickness`, on 11 validation cases only, Dice equivalence 0.001 then lower HD95 then smaller mm. Test cannot change a validation lock.

## Execution

```bash
CODE=/home/wusi/SAM2/MyTrain/MyCodes/Rectal/R-20260922/binary_mask/skeleton_thickness
RUN=/home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260922/binary_mask/skeleton_thickness
PY=/home/wusi/miniconda3/envs/sam2/bin/python
$PY $CODE/preflight_skeleton.py --data-root /home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask --marker $RUN/PREFLIGHT_PASS
$PY $CODE/schedule_training.py --run-root $RUN --reserve-gpus 6 --max-parallel 4
$PY $CODE/run_validation_inference.py --run-root $RUN --gpus 0 1 2 3
for K in pos neg; do $PY $CODE/select_branch_thickness_validation.py --kind $K --data-root /home/wusi/SAM2/MyTrain/SAM2data/Rectal/R-20260720/Prompt_mask --split-json $RUN/checkpoints/$K/t00mm/split.json --cache-root $RUN/probability_cache_validation --metrics-root $RUN/metrics; done
$PY $CODE/plot_validation_profiles.py --metrics-root $RUN/metrics
$PY $CODE/run_test_inference.py --run-root $RUN --gpus 0 1 2 3
```

Then execute `fusion_test/run_fusion_test.py` with the test cache root, validation metric root and output root. It saves per patient POS, NEG, Direct, and final fused NIfTI with source image geometry. Direct uses `(N | POS_prompt) & ~(N & NEG_prompt)` and fusion uses `POS_pred & ~(N & ~NEG_pred)`.
