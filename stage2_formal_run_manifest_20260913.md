# Stage-2 fold-0 formal runs

Objective: compare SAM2 and MedSAM2 using the same native multi-frame,
multi-point persistent-state correction protocol.

Immutable protocol: initial K masks registered once; one residual click per
round; per-frame point histories; native non-conditioning correction; T sampled
uniformly from 0 through 5; terminal truncated replay; validation K=3 across
two fixed placements with sequential D0 through D5, selected by mean D0--D5.

SAM2: GPU 4, tmux `sam2_stage2_formal_fold0`, output
`/home/wusi/SAM2/MyTrain/SAM2data/Eso/20260909_CTV/Stage2-point/TrainResults/Stage2_T0_5/fold_0`.

MedSAM2: GPU 5, tmux `medsam2_stage2_formal_fold0`, output
`/home/wusi/MedSAM2/MyResults/Eso/20260909_CTV/Stage2-point/TrainResults/Stage2_T0_5/fold_0`.

Both runs use 100 epochs, validation at epoch 1 and every ten epochs, and
persist `latest.pth` for resumption.  `Stage1` and `Stage2-point_full` are not
part of cleanup or overwrite targets.
