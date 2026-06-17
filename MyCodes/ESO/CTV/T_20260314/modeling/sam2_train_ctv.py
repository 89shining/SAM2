import torch
from training.model.sam2 import SAM2Train


class SAM2TrainCTV(SAM2Train):
    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        支持 batch>1：
        - 固定使用 GT mask 作为 prompt
        - 不使用 point / box / iterative correction
        - 允许 batch 内不同 object 有不同的 prompt_frame_idx
        """

        # 1) 整理每一帧的 GT mask: [O,H,W] -> [O,1,H,W]
        gt_masks_per_frame = {
            frame_idx: masks.unsqueeze(1)
            for frame_idx, masks in enumerate(input.masks)
        }

        num_frames = input.num_frames
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["num_frames"] = num_frames

        # 2) 强制只用 mask prompt
        backbone_out["use_pt_input"] = False
        backbone_out["point_inputs_per_frame"] = {}

        # 3) 读取每个 object 的 prompt frame
        # input.metadata.prompt_frame_idx shape: [T, O]
        # 因为同一个 object 在所有 T 上存的是同一个 prompt_frame_idx，所以取第 0 行即可
        obj_prompt_frames = input.metadata.prompt_frame_idx[0]  # [O]

        # 安全检查
        assert obj_prompt_frames.dim() == 1, f"Expected [O], got {obj_prompt_frames.shape}"
        for o in range(obj_prompt_frames.shape[0]):
            pf = int(obj_prompt_frames[o].item())
            assert 0 <= pf < num_frames, f"Object {o}: prompt_frame_idx={pf} out of range [0, {num_frames-1}]"

        # 4) batch 内所有出现过的 prompt frame 都作为初始条件帧
        init_cond_frames = sorted(torch.unique(obj_prompt_frames).tolist())

        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames)
            if t not in init_cond_frames
        ]

        # 5) 为每个 init_cond_frame 构造 mask prompt
        # 形式仍保持 [O,1,H,W]，但只有“该在这帧提示”的 object 放真实 mask，其余 object 放全0
        backbone_out["mask_inputs_per_frame"] = {}

        for t in init_cond_frames:
            gt_t = gt_masks_per_frame[t]  # [O,1,H,W]
            prompt_t = torch.zeros_like(gt_t)  # [O,1,H,W]

            for o in range(obj_prompt_frames.shape[0]):
                if int(obj_prompt_frames[o].item()) == t:
                    prompt_t[o] = gt_t[o]

            backbone_out["mask_inputs_per_frame"][t] = prompt_t

        # 6) 不使用 correction points
        backbone_out["frames_to_add_correction_pt"] = []

        # # 调试输出
        # print(f"[SAM2TrainCTV] num_frames = {num_frames}")
        # print(f"[SAM2TrainCTV] obj_prompt_frames = {obj_prompt_frames.tolist()}")
        # print(f"[SAM2TrainCTV] init_cond_frames = {init_cond_frames}")

        return backbone_out