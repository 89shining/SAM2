"""Explicit access to the locked Stage-1 implementation used by Stage 2.

Stage 2 intentionally reuses Stage-1's dataset/model construction and immutable
prompt plan. The sibling path is resolved rather than guessed from PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

STAGE2_DIR = Path(__file__).resolve().parent
STAGE1_DIR = STAGE2_DIR.parent / "Stage1-mask"
if not STAGE1_DIR.is_dir():
    raise FileNotFoundError(f"Required sibling Stage1-mask directory missing: {STAGE1_DIR}")
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from experiment_core import (  # noqa: E402,F401
    DiceBCELoss,
    feasible_prompt_counts,
    positive_slice_indices,
    sample_train_prompt_indices,
    stratified_spaced_prompt_indices,
)
from io_utils import (  # noqa: E402,F401
    DEFAULT_DATA_ROOT,
    DEFAULT_INIT_CKPT,
    DEFAULT_MODEL_CFG,
    DEFAULT_SPLIT_PATH,
    RectalCTVVolumeDataset,
    build_model,
    build_optimizer,
    build_scheduler,
    enable_image_encoder_activation_checkpointing,
    list_patient_dirs,
    load_checkpoint,
    make_or_load_splits,
    save_checkpoint,
    set_global_seed,
)
from bidirectional_tracking import bidirectional_outputs  # noqa: E402,F401

