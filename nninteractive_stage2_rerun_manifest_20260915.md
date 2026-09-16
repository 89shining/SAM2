# nnInteractive Stage-2 fold-0 regeneration

Purpose: regenerate the deleted fold-0 Stage-2 result with the existing code,
Stage-1 initialization, fixed seed, and standard validation protocol.

Environment: `nninteractive`; GPU 5; tmux session
`nninteractive_stage2_formal_fold0`.

Command: `python train.py --fold 0 --epochs 100` from
`/home/wusi/nnInteractive/MyCodes/ESO/CTV/T-20260909/Stage2-point`.

Output: `/home/wusi/nnInteractive/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults/Workflow_K3_T0_5/fold_0`.

The run writes `latest.pth` for safe resumption and `best.pth` selected by the
workflow metric. Existing Stage-1 results are read-only inputs.
