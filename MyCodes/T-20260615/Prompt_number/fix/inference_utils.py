#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@torch.no_grad()
def bidirectional_predict_with_gt_mask_prompts(
    predictor,
    frame_dir: Path,
    gt_zyx: np.ndarray,
    prompt_frames: Sequence[int],
    obj_id: int = 1,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_frames = list(dict.fromkeys(int(x) for x in prompt_frames))
    z, h, w = gt_zyx.shape
    prompt_frames = [max(0, min(t, z - 1)) for t in prompt_frames]

    def run_one_direction(reverse: bool) -> np.ndarray:
        state = predictor.init_state(video_path=str(frame_dir))
        predictor.reset_state(state)
        for t in prompt_frames:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=t,
                obj_id=obj_id,
                mask=(gt_zyx[t] > 0).astype(np.uint8),
            )

        probs = np.full((z, h, w), np.nan, dtype=np.float32)
        for fidx, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
            for i, oid in enumerate(obj_ids):
                if int(oid) == int(obj_id):
                    probs[int(fidx)] = torch.sigmoid(logits[i]).detach().cpu().numpy().astype(np.float32)
                    break
        return probs

    prob_f = run_one_direction(reverse=False)
    prob_b = run_one_direction(reverse=True)
    prob = np.nanmean(np.stack([prob_f, prob_b], axis=0), axis=0)

    missing = np.isnan(prob)
    if missing.any():
        prob[missing] = np.nan_to_num(prob_f[missing], nan=0.0)
        still_missing = np.isnan(prob)
        prob[still_missing] = np.nan_to_num(prob_b[still_missing], nan=0.0)

    pred = (prob >= float(threshold)).astype(np.uint8)
    return pred, prob
