import torch

from training.model.sam2 import SAM2Train


class SAM2TrainUpperLowerMiddleRuleMask(SAM2Train):
    """
    Train/eval with mask prompts only:
    - For each object, automatically find GT-positive temporal bounds (lower/upper).
    - Middle prompt follows rule_mask/mask_prompt_rule.py middle rule:
      choose geometric middle among GT-positive middle layers (lower < z < upper);
      if no middle candidate, fallback to lower.
    - Iterative prompt schedule:
      stage-1 give upper + lower as initial conditioning prompts,
      stage-2 add middle prompt on top of current tracking state.
    - Disable point/box prompts and iterative correction clicks.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                prob_to_use_pt_input_for_train=0.0,
                prob_to_use_pt_input_for_eval=0.0,
                prob_to_use_box_input_for_train=0.0,
                prob_to_use_box_input_for_eval=0.0,
                prob_to_sample_from_gt_for_train=0.0,
                num_frames_to_correct_for_train=1,
                num_frames_to_correct_for_eval=1,
                rand_frames_to_correct_for_train=False,
                rand_frames_to_correct_for_eval=False,
                add_all_frames_to_correct_as_cond=False,
                num_correction_pt_per_frame=0,
                rand_init_cond_frames_for_train=False,
                rand_init_cond_frames_for_eval=False,
            )
        )
        super().__init__(*args, **kwargs)
        self.enable_middle_prompt = False

    def set_middle_prompt_enabled(self, enabled: bool):
        self.enable_middle_prompt = bool(enabled)

    @staticmethod
    def _choose_middle_rule(pos_t: torch.Tensor, lower: int, upper: int) -> int:
        middle_candidates = [int(z) for z in pos_t.tolist() if lower < int(z) < upper]
        if len(middle_candidates) == 0:
            return int(lower)
        middle_candidates = sorted(middle_candidates)
        return int(middle_candidates[len(middle_candidates) // 2])

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        gt_masks_per_frame = {
            frame_idx: masks.unsqueeze(1)
            for frame_idx, masks in enumerate(input.masks)
        }
        num_frames = input.num_frames

        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["num_frames"] = num_frames
        backbone_out["use_pt_input"] = False
        backbone_out["point_inputs_per_frame"] = {}
        backbone_out["frames_to_add_correction_pt"] = []

        # input.masks: [T, O, H, W]
        masks_tohw = input.masks
        if masks_tohw.ndim != 4:
            raise ValueError(f"Expected input.masks to be [T, O, H, W], got {masks_tohw.shape}")

        t_dim, o_dim = masks_tohw.shape[:2]
        if t_dim != num_frames:
            raise ValueError(f"num_frames mismatch: {num_frames} vs {t_dim}")

        lower_ids = []
        upper_ids = []
        middle_ids = []
        for obj_idx in range(o_dim):
            per_t_has_fg = masks_tohw[:, obj_idx].flatten(1).any(dim=1)
            pos_t = torch.nonzero(per_t_has_fg, as_tuple=False).flatten()
            if pos_t.numel() == 0:
                lower = int(start_frame_idx)
                upper = int(start_frame_idx)
                middle = int(start_frame_idx)
            else:
                lower = int(pos_t.min().item())
                upper = int(pos_t.max().item())
                if self.enable_middle_prompt:
                    middle = self._choose_middle_rule(pos_t, lower, upper)
                else:
                    middle = None

            lower_ids.append(lower)
            upper_ids.append(upper)
            middle_ids.append(middle)

        init_cond_frames = sorted(set(lower_ids + upper_ids))
        backbone_out["init_cond_frames"] = init_cond_frames
        if self.enable_middle_prompt:
            middle_prompt_frames = sorted(
                set([m for m in middle_ids if m is not None]) - set(init_cond_frames)
            )
            remaining_frames = [
                t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
            ]
            remaining_wo_middle = [t for t in remaining_frames if t not in middle_prompt_frames]
            backbone_out["frames_not_in_init_cond"] = middle_prompt_frames + remaining_wo_middle
        else:
            backbone_out["frames_not_in_init_cond"] = [
                t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
            ]

        backbone_out["mask_inputs_per_frame"] = {}
        if self.enable_middle_prompt:
            prompt_frames = sorted(set(init_cond_frames + [m for m in middle_ids if m is not None]))
        else:
            prompt_frames = init_cond_frames
        for t in prompt_frames:
            gt_t = gt_masks_per_frame[t]  # [O, 1, H, W]
            prompt_t = torch.zeros_like(gt_t)
            for o in range(o_dim):
                if lower_ids[o] == t or upper_ids[o] == t or (
                    self.enable_middle_prompt and middle_ids[o] is not None and middle_ids[o] == t
                ):
                    prompt_t[o] = gt_t[o]
            backbone_out["mask_inputs_per_frame"][t] = prompt_t

        return backbone_out
