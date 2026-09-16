# SAM2 Stage2: official-style multi-frame correction

This directory is the adapted SAM2 Stage2 experiment.  It supersedes the
former full-history replay code; its old `TrainResults` directory was removed
before this implementation was installed.

Initial GT masks are registered once and are the only conditioning frames.
At correction round `t`, the selected non-initial frame retains its own point
history and current continuous low-resolution logits as the native mask prior.
The correction remains a non-conditioning output rather than becoming a
global memory anchor: this is SAM2's default
`add_all_frames_to_correct_as_cond=False` behavior.

Training retains `K=1..5`, `T=0..5`, the same correction oracle, BF16,
activation checkpointing, full-network fine-tuning, loss, and validation plan.
For `T>=1`, historical state transitions run under `torch.no_grad()` and only
the terminal native correction plus its directional continuation is differentiated.
