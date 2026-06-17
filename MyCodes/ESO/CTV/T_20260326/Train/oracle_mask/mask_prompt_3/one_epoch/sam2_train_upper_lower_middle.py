import torch

from training.model.sam2 import SAM2Train


class SAM2TrainUpperLowerMiddleMask(SAM2Train):
    """
    Train/eval with mask prompts only:
    - For each object, automatically find GT-positive temporal bounds (lower/upper).
    - Read patient-level middle prompt index from external table (mask_prompt_3 result).
    - Use GT masks on upper + lower + middle frames as conditioning prompts.
    - Disable point/box prompts and iterative correction clicks.
    """

    def __init__(self, *args, middle_prompt_by_video_id=None, **kwargs):
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
        self.middle_prompt_by_video_id = {
            int(k): int(v) for k, v in (middle_prompt_by_video_id or {}).items()
        }

    @staticmethod
    def _choose_fallback_middle(pos_t: torch.Tensor, lower: int, upper: int) -> int:
        middle_candidates = [int(z) for z in pos_t.tolist() if lower < int(z) < upper]
        if len(middle_candidates) == 0:
            return lower
        middle_candidates = sorted(middle_candidates)
        return int(middle_candidates[len(middle_candidates) // 2])

    @staticmethod
    def _valid_mid_for_object(gt_obj_t_hw: torch.Tensor, mid: int, lower: int, upper: int) -> int:
        t_dim = gt_obj_t_hw.shape[0]
        mid = max(0, min(int(mid), t_dim - 1))
        if bool(gt_obj_t_hw[mid].any()):
            return mid

        # If external id is invalid for this object, fallback to lower (always positive by construction).
        if bool(gt_obj_t_hw[lower].any()):
            return int(lower)
        if bool(gt_obj_t_hw[upper].any()):
            return int(upper)
        return int(mid)

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

        # [T, O, 3] -> use t=0 because video_id is constant over time for one object.
        # unique_objects_identifier[..., 0] stores original video_id.
        obj_video_ids = input.metadata.unique_objects_identifier[0, :, 0].to(torch.long)

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

                video_id = int(obj_video_ids[obj_idx].item())
                external_mid = self.middle_prompt_by_video_id.get(video_id)
                if external_mid is None:
                    external_mid = self._choose_fallback_middle(pos_t, lower, upper)

                middle = self._valid_mid_for_object(
                    gt_obj_t_hw=masks_tohw[:, obj_idx],
                    mid=int(external_mid),
                    lower=lower,
                    upper=upper,
                )

            lower_ids.append(lower)
            upper_ids.append(upper)
            middle_ids.append(middle)

        init_cond_frames = sorted(set(lower_ids + upper_ids + middle_ids))
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]

        backbone_out["mask_inputs_per_frame"] = {}
        for t in init_cond_frames:
            gt_t = gt_masks_per_frame[t]  # [O, 1, H, W]
            prompt_t = torch.zeros_like(gt_t)
            for o in range(o_dim):
                if lower_ids[o] == t or upper_ids[o] == t or middle_ids[o] == t:
                    prompt_t[o] = gt_t[o]
            backbone_out["mask_inputs_per_frame"][t] = prompt_t

        return backbone_out
