# R-20260922 / binary_mask final test

1. Run `preflight_skeleton.py` once; it must create `PREFLIGHT_PASS`.
2. Run `schedule_training.py` (it is intentionally training-only).
3. For each branch, infer validation from every `best.pth` using `infer_skeleton_thickness.py --subset validation --segment-fraction 0.5 --prompt-fraction 0.5`; then run `select_branch_thickness_validation.py` and `plot_validation_profiles.py`.
4. The selection script locks `epoch -> m*(thickness) -> thickness*` from validation alone.
5. Infer test for every thickness at axial fractions `0, .25, .5, .75, 1`; retain each cached prompt mask.
6. Run `run_fusion_test.py`. It only reads validation locks and never reads test metrics to choose a parameter. It writes per-patient POS, NEG, Direct, and fused NIfTI predictions with original image geometry.

`Direct = (N | POS_prompt) & ~(N & NEG_prompt)` and final fusion is `POS_pred & ~(N & ~NEG_pred)`.
