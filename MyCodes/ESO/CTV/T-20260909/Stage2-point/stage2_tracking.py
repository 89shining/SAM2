"""Official-style multi-frame SAM2 state tracking for Stage-2 correction.

Initial mask-prompt frames are the only conditioning frames.  A later outer
correction may occur on any non-initial frame; it retains that frame's point
history and continuous low-resolution mask prior, but remains a native
non-conditioning correction output.  This follows SAM2's default
``add_all_frames_to_correct_as_cond=False`` semantics rather than promoting
every click to a global memory anchor.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from point_clicker import CorrectionPoint


def _normal_detached(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Copy a tensor into ordinary (non-inference) storage without a gradient."""
    value = value.detach()
    with torch.inference_mode(False):
        result = torch.empty(value.shape, device=value.device, dtype=dtype or value.dtype)
        result.copy_(value)
    if result.is_inference():
        raise RuntimeError("Could not convert inference tensor to normal storage")
    return result


def _capture_decoder(captured: dict):
    def hook(_module, _inputs, output):
        captured["multimasks"] = output[0]
        captured["ious"] = output[1]
    return hook


def _attach_raw_logits(core_model, output: dict, captured: dict) -> None:
    """Attach differentiable decoder logits without replacing SAM2 state masks."""
    if "multimasks" not in captured:
        raise RuntimeError("SAM decoder output was not captured")
    masks, ious = captured["multimasks"].float(), captured["ious"]
    if masks.shape[1] > 1:
        index = torch.argmax(ious, dim=-1)
        batch_index = torch.arange(masks.shape[0], device=masks.device)
        low_res = masks[batch_index, index].unsqueeze(1)
    else:
        low_res = masks
    output["stage2_raw_pred_masks"] = low_res
    output["stage2_raw_pred_masks_high_res"] = F.interpolate(
        low_res,
        size=(core_model.image_size, core_model.image_size),
        mode="bilinear",
        align_corners=False,
    )


def _point_input(points: Sequence[CorrectionPoint], device: torch.device) -> dict:
    if not points:
        raise ValueError("A point-conditioned frame needs at least one point")
    return {
        "point_coords": torch.tensor(
            [[[point.x, point.y] for point in points]],
            dtype=torch.float32,
            device=device,
        ),
        "point_labels": torch.tensor(
            [[1 if point.label > 0 else 0 for point in points]],
            dtype=torch.int64,
            device=device,
        ),
    }


def _prepare_features(core_model, backbone: dict):
    if backbone.get("backbone_fpn") is None:
        raise RuntimeError("Invalid cached full-volume backbone features")
    return core_model._prepare_backbone_features(backbone)[1:]


def _normal_detached_tree(value):
    """Deep-copy a native no-grad state into ordinary detached tensor storage."""
    if isinstance(value, torch.Tensor):
        return _normal_detached(value)
    if isinstance(value, dict):
        return {key: _normal_detached_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normal_detached_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normal_detached_tree(item) for item in value)
    return value


class NativeDirectionState:
    """One autograd-compatible SAM2 direction with persistent conditioning memory."""

    def __init__(
        self,
        core_model,
        batch,
        initial_mask_frames: Sequence[int],
        reverse: bool,
        base_backbone_out: dict | None = None,
        trace: list[dict] | None = None,
        auto_propagate: bool = True,
    ) -> None:
        self.model = core_model
        self.batch = batch
        self.reverse = bool(reverse)
        self.trace_direction = "reverse" if self.reverse else "forward"
        self.trace = trace
        self.initial_frames = tuple(sorted({int(frame) for frame in initial_mask_frames}))
        if not self.initial_frames:
            raise ValueError("Native Stage2 state needs at least one initial mask frame")
        self.initial_set = set(self.initial_frames)
        self.point_history: dict[int, list[CorrectionPoint]] = defaultdict(list)
        self.cond_frames = set(self.initial_frames)
        self.output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.outputs_by_frame: dict[int, dict] = {}
        self._cached_features = _prepare_features(core_model, base_backbone_out) if base_backbone_out else None
        self._register_initial_masks()
        if auto_propagate:
            self.propagate()

    def _features(self, frame: int):
        image_ids = self.batch.flat_obj_to_img_idx[frame]
        if self._cached_features is not None:
            vision_feats, vision_pos, feat_sizes = self._cached_features
            return (
                [feature[:, image_ids] for feature in vision_feats],
                [position[:, image_ids] for position in vision_pos],
                feat_sizes,
            )
        _, feats, pos, sizes = self.model._prepare_backbone_features_per_frame(
            self.batch.flat_img_batch, image_ids
        )
        return feats, pos, sizes

    def _run(
        self,
        frame: int,
        *,
        is_initial_mask: bool = False,
        point_inputs: dict | None = None,
        previous_logits: torch.Tensor | None = None,
        bypass_memory_attention: bool = False,
    ) -> dict:
        feats, pos, feat_sizes = self._features(frame)
        mask_inputs = (
            _normal_detached(self.batch.masks[frame].unsqueeze(1))
            if is_initial_mask
            else None
        )
        gt_masks = (
            _normal_detached(self.batch.masks[frame].unsqueeze(1))
            if is_initial_mask
            else None
        )
        captured: dict = {}
        hook = None
        if not is_initial_mask:
            hook = self.model.sam_mask_decoder.register_forward_hook(_capture_decoder(captured))
        original_memory_prepare = None
        if bypass_memory_attention:
            # Local correction uses the image feature itself, not the
            # direction-specific memory-conditioned feature.  This retains
            # the ordinary correction path (point prompt, continuous mask
            # prior, is_init_cond_frame=False and memory encoding afterwards)
            # while decoupling only the clicked-frame SAM-head input.
            original_memory_prepare = self.model._prepare_memory_conditioned_features

            def unconditioned_feature(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse=False,
            ):
                del (
                    frame_idx, is_init_cond_frame, current_vision_pos_embeds,
                    output_dict, num_frames, track_in_reverse,
                )
                feature = current_vision_feats[-1]
                batch_size = feature.size(1)
                channels = feature.size(2)
                height, width = feat_sizes[-1]
                return feature.permute(1, 2, 0).view(batch_size, channels, height, width)

            self.model._prepare_memory_conditioned_features = unconditioned_feature
        try:
            current = self.model.track_step(
                frame_idx=frame,
                # Only clinician-supplied initial GT-mask frames take SAM2's
                # initialization branch. Multi-frame correction frames retain
                # their native non-conditioning state and refine prior logits.
                is_init_cond_frame=frame in self.initial_set,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                gt_masks=gt_masks,
                frames_to_add_correction_pt=[],
                output_dict=self.output_dict,
                num_frames=int(self.batch.num_frames),
                track_in_reverse=self.reverse,
                prev_sam_mask_logits=previous_logits,
            )
        finally:
            if original_memory_prepare is not None:
                self.model._prepare_memory_conditioned_features = original_memory_prepare
            if hook is not None:
                hook.remove()
        if not is_initial_mask:
            # Keep current["pred_masks"] and its memory representation exactly as
            # returned by native SAM2.  Raw decoder logits are only an auxiliary
            # differentiable view for the Stage2 segmentation loss.
            _attach_raw_logits(self.model, current, captured)
        else:
            # Initial-mask slices are excluded from the segmentation loss, but
            # keeping the same field on every frame makes batch loss assembly
            # explicit and avoids a special-case tensor stack.
            current["stage2_raw_pred_masks"] = current["pred_masks"]
            current["stage2_raw_pred_masks_high_res"] = current["pred_masks_high_res"]
        if self.trace is not None:
            self.trace.append(
                {
                    "direction": self.trace_direction,
                    "frame": int(frame),
                    "is_initial_mask_frame": bool(is_initial_mask),
                    "is_init_cond_frame": frame in self.initial_set,
                    "is_conditioning_output": frame in self.cond_frames,
                    "is_point_frame": point_inputs is not None,
                    "mask_inputs": bool(is_initial_mask),
                    "gt_masks": bool(is_initial_mask),
                    "point_inputs": point_inputs is not None,
                    "point_count_on_frame": 0 if point_inputs is None else int(point_inputs["point_labels"].shape[1]),
                    "uses_previous_logits": previous_logits is not None,
                    "bypasses_memory_attention": bool(bypass_memory_attention),
                    "prev_sam_mask_logits_shape": (
                        None if previous_logits is None else list(previous_logits.shape)
                    ),
                }
            )
        bucket = "cond_frame_outputs" if frame in self.cond_frames else "non_cond_frame_outputs"
        self.output_dict[bucket][frame] = current
        self.outputs_by_frame[frame] = current
        return current

    def _register_initial_masks(self) -> None:
        # This is the sole pathway that supplies ground-truth masks to SAM2.
        for frame in sorted(self.initial_set, reverse=self.reverse):
            self._run(frame, is_initial_mask=True)

    def propagate(self) -> None:
        """Refresh only non-conditioning frames from the persistent state."""
        self.output_dict["non_cond_frame_outputs"].clear()
        for frame in list(self.outputs_by_frame):
            if frame not in self.cond_frames:
                self.outputs_by_frame.pop(frame)
        order = (
            range(int(self.batch.num_frames) - 1, -1, -1)
            if self.reverse
            else range(int(self.batch.num_frames))
        )
        for frame in order:
            if frame not in self.cond_frames:
                self._run(frame)
        missing = set(range(int(self.batch.num_frames))) - set(self.outputs_by_frame)
        if missing:
            raise RuntimeError(f"Propagation omitted frames: {sorted(missing)}")

    def add_click(
        self,
        click: CorrectionPoint,
        previous_logits_override: torch.Tensor | None = None,
        propagate: bool = True,
        bypass_memory_attention: bool = False,
    ) -> None:
        frame = int(click.z)
        if frame in self.initial_set:
            raise RuntimeError("Correction points cannot lie on initial-mask slices")
        if frame not in self.outputs_by_frame and previous_logits_override is None:
            raise RuntimeError(f"No previous prediction for correction frame {frame}")
        self.point_history[frame].append(click)
        # SAM2's native refinement uses the current continuous low-res logits,
        # not a reconstructed hard volume, together with same-frame point history.
        previous_logits = (
            torch.clamp(self.outputs_by_frame[frame]["pred_masks"], -32.0, 32.0)
            if previous_logits_override is None
            else torch.clamp(previous_logits_override, -32.0, 32.0)
        )
        self.cond_frames.add(frame)
        self.output_dict["non_cond_frame_outputs"].pop(frame, None)
        self._run(
            frame,
            point_inputs=_point_input(self.point_history[frame], previous_logits.device),
            previous_logits=previous_logits,
            bypass_memory_attention=bypass_memory_attention,
        )
        if propagate:
            self.propagate()

    def add_click_as_non_cond(self, click: CorrectionPoint) -> None:
        """Apply one persistent multi-frame correction without making it cond.

        This is the task-level counterpart of SAM2's default correction-frame
        handling: only initial prompt masks are conditioning frames.  Same
        frame clicks accumulate their native point history; subsequent frames
        are refreshed directionally while previously corrected non-cond frames
        remain available as persistent tracking outputs.
        """
        frame = int(click.z)
        if frame in self.initial_set:
            raise RuntimeError("Correction points cannot lie on initial-mask slices")
        if frame not in self.outputs_by_frame:
            raise RuntimeError(f"No previous prediction for correction frame {frame}")
        if frame in self.cond_frames:
            raise RuntimeError("Only initial masks may be conditioning frames")

        self.point_history[frame].append(click)
        previous_logits = torch.clamp(self.outputs_by_frame[frame]["pred_masks"], -32.0, 32.0)
        self.output_dict["non_cond_frame_outputs"].pop(frame, None)
        self._run(
            frame,
            point_inputs=_point_input(self.point_history[frame], previous_logits.device),
            previous_logits=previous_logits,
        )

        # Preserve any earlier corrected frame.  It is re-used by native
        # tracking as an existing non-conditioning output, not reinterpreted
        # as a new global conditioning memory.
        order = range(frame - 1, -1, -1) if self.reverse else range(frame + 1, int(self.batch.num_frames))
        for next_frame in order:
            if next_frame in self.cond_frames or next_frame in self.point_history:
                continue
            self._run(next_frame)

    def detached_terminal_snapshot(self) -> "NativeDirectionState":
        """Return S(t-1) as normal detached state for one terminal update.

        The clone retains native initial/correction memories, current native
        logits, and same-frame point history, but intentionally discards the
        preceding autograd graph and any cached no-grad image features.  The
        caller then performs exactly one final correction and propagation with
        gradients enabled.
        """
        snapshot = object.__new__(NativeDirectionState)
        snapshot.model = self.model
        snapshot.batch = self.batch
        snapshot.reverse = self.reverse
        snapshot.trace_direction = self.trace_direction
        snapshot.trace = self.trace
        snapshot.initial_frames = self.initial_frames
        snapshot.initial_set = set(self.initial_set)
        snapshot.point_history = defaultdict(
            list, {frame: list(points) for frame, points in self.point_history.items()}
        )
        snapshot.cond_frames = set(self.cond_frames)
        snapshot.output_dict = {
            "cond_frame_outputs": _normal_detached_tree(self.output_dict["cond_frame_outputs"]),
            "non_cond_frame_outputs": _normal_detached_tree(self.output_dict["non_cond_frame_outputs"]),
        }
        snapshot.outputs_by_frame = {
            **snapshot.output_dict["cond_frame_outputs"],
            **snapshot.output_dict["non_cond_frame_outputs"],
        }
        # The terminal round must recompute its image features with autograd;
        # only the persistent interaction state is inherited from S(t-1).
        snapshot._cached_features = None
        return snapshot

    def propagation_view(self, reverse: bool, *, detach: bool = True) -> "NativeDirectionState":
        """Create a directional propagation view from shared conditioning outputs.

        ``detach=False`` is used for the differentiable terminal transition:
        both propagation directions then consume the exact same canonical
        corrected memory.  ``detach=True`` remains useful for diagnostic and
        no-grad state construction.
        """
        if detach:
            view = self.detached_terminal_snapshot()
        else:
            view = object.__new__(NativeDirectionState)
            view.model = self.model
            view.batch = self.batch
            view.trace = self.trace
            view.initial_frames = self.initial_frames
            view.initial_set = set(self.initial_set)
            view.point_history = defaultdict(
                list, {frame: list(points) for frame, points in self.point_history.items()}
            )
            view.cond_frames = set(self.cond_frames)
            view.output_dict = {
                "cond_frame_outputs": self.output_dict["cond_frame_outputs"],
                "non_cond_frame_outputs": {},
            }
            view.outputs_by_frame = dict(view.output_dict["cond_frame_outputs"])
            view._cached_features = self._cached_features
        view.reverse = bool(reverse)
        view.trace_direction = "reverse" if view.reverse else "forward"
        view.output_dict["non_cond_frame_outputs"] = {}
        view.outputs_by_frame = dict(view.output_dict["cond_frame_outputs"])
        view.propagate()
        return view

    def outputs(self) -> list[dict]:
        return [
            {key: value for key, value in self.outputs_by_frame[frame].items() if key != "obj_ptr"}
            for frame in range(int(self.batch.num_frames))
        ]


class NativeBidirectionalState:
    """Persistent forward/reverse states used for one Stage2 interaction episode."""

    def __init__(
        self,
        core_model,
        batch,
        initial_mask_frames: Sequence[int],
        base_backbone_out: dict | None = None,
        trace: list[dict] | None = None,
    ) -> None:
        self.forward = NativeDirectionState(
            core_model, batch, initial_mask_frames, False, base_backbone_out, trace
        )
        self.reverse = NativeDirectionState(
            core_model, batch, initial_mask_frames, True, base_backbone_out, trace
        )

    def add_click(self, click: CorrectionPoint) -> None:
        self.forward.add_click(click)
        self.reverse.add_click(click)

    def detached_terminal_snapshot(self) -> "NativeBidirectionalState":
        """Deep-detached persistent state for truncated terminal BPTT."""
        snapshot = object.__new__(NativeBidirectionalState)
        snapshot.forward = self.forward.detached_terminal_snapshot()
        snapshot.reverse = self.reverse.detached_terminal_snapshot()
        return snapshot

    @staticmethod
    def fused_low_res_prior(forward: NativeDirectionState, reverse: NativeDirectionState, frame: int) -> torch.Tensor:
        """Canonical continuous correction prior shown by the fused prediction."""
        probability = 0.5 * (
            torch.sigmoid(forward.outputs_by_frame[frame]["pred_masks"])
            + torch.sigmoid(reverse.outputs_by_frame[frame]["pred_masks"])
        )
        return torch.logit(probability.clamp(1e-4, 1.0 - 1e-4))

    def outputs(self) -> list[dict]:
        forward, reverse = self.forward.outputs(), self.reverse.outputs()
        fused = []
        for out_f, out_b in zip(forward, reverse):
            native_probability = 0.5 * (
                torch.sigmoid(out_f["pred_masks_high_res"])
                + torch.sigmoid(out_b["pred_masks_high_res"])
            )
            raw_probability = 0.5 * (
                torch.sigmoid(out_f["stage2_raw_pred_masks_high_res"])
                + torch.sigmoid(out_b["stage2_raw_pred_masks_high_res"])
            )
            out = dict(out_f)
            out["pred_masks_high_res"] = torch.logit(
                native_probability.clamp(1e-4, 1.0 - 1e-4)
            )
            out["stage2_raw_pred_masks_high_res"] = torch.logit(
                raw_probability.clamp(1e-4, 1.0 - 1e-4)
            )
            out["stage2_forward_object_score_logits"] = out_f["multistep_object_score_logits"][-1]
            out["stage2_reverse_object_score_logits"] = out_b["multistep_object_score_logits"][-1]
            fused.append(out)
        return fused


class OfficialMultiFrameBidirectionalState(NativeBidirectionalState):
    """Two native directional SAM2 states with multi-frame non-cond clicks.

    The external Stage-2 oracle contributes exactly one point per outer round.
    SAM2's persistent state supplies the per-frame point accumulation and
    previous-logit refinement; no initial mask or full prompt history is
    reconstructed at each round.
    """

    def add_click(self, click: CorrectionPoint) -> None:
        self.forward.add_click_as_non_cond(click)
        self.reverse.add_click_as_non_cond(click)

    def detached_terminal_snapshot(self) -> "OfficialMultiFrameBidirectionalState":
        snapshot = object.__new__(OfficialMultiFrameBidirectionalState)
        snapshot.forward = self.forward.detached_terminal_snapshot()
        snapshot.reverse = self.reverse.detached_terminal_snapshot()
        return snapshot


class MemoryDecoupledBidirectionalState:
    """One local correction state with shared memory and two propagation views.

    Initial GT masks are registered once.  Every user correction is decoded
    once from image-only features with the fused continuous prior, then encoded
    into a single canonical conditioning-memory state.  Forward and reverse
    paths only propagate from that shared corrected memory; they never decode
    the click independently.
    """

    def __init__(
        self,
        core_model,
        batch,
        initial_mask_frames: Sequence[int],
        base_backbone_out: dict | None = None,
        trace: list[dict] | None = None,
    ) -> None:
        self.canonical = NativeDirectionState(
            core_model,
            batch,
            initial_mask_frames,
            False,
            base_backbone_out,
            trace,
            auto_propagate=False,
        )
        self.canonical.trace_direction = "canonical"
        # P0 is generated only by the two views below.  The canonical object
        # owns conditioning outputs/memories, not a directional trajectory.
        self.canonical.output_dict["non_cond_frame_outputs"] = {}
        self.canonical.outputs_by_frame = dict(self.canonical.output_dict["cond_frame_outputs"])
        self._refresh_propagation_views(detach=False)

    def _refresh_propagation_views(self, *, detach: bool) -> None:
        self.forward = self.canonical.propagation_view(False, detach=detach)
        self.reverse = self.canonical.propagation_view(True, detach=detach)

    def propagate_from_canonical(self, *, detach: bool = False) -> None:
        """Rebuild both directional predictions from the shared memory state."""
        self._refresh_propagation_views(detach=detach)

    def add_click(self, click: CorrectionPoint) -> None:
        # This is the fused continuous low-resolution counterpart of the
        # high-resolution prediction shown to the oracle/user.  It is fed
        # once to the single local correction decode.
        prior = NativeBidirectionalState.fused_low_res_prior(
            self.forward, self.reverse, int(click.z)
        )
        self.canonical.add_click(
            click,
            previous_logits_override=prior,
            propagate=False,
            bypass_memory_attention=True,
        )
        self._refresh_propagation_views(detach=False)

    def detached_terminal_snapshot(self) -> "MemoryDecoupledBidirectionalState":
        """Deep-detach S(t-1) before one trainable local transition."""
        snapshot = object.__new__(MemoryDecoupledBidirectionalState)
        snapshot.canonical = self.canonical.detached_terminal_snapshot()
        snapshot.forward = self.forward.detached_terminal_snapshot()
        snapshot.reverse = self.reverse.detached_terminal_snapshot()
        return snapshot

    def outputs(self) -> list[dict]:
        forward, reverse = self.forward.outputs(), self.reverse.outputs()
        fused = []
        for out_f, out_b in zip(forward, reverse):
            native_probability = 0.5 * (
                torch.sigmoid(out_f["pred_masks_high_res"])
                + torch.sigmoid(out_b["pred_masks_high_res"])
            )
            raw_probability = 0.5 * (
                torch.sigmoid(out_f["stage2_raw_pred_masks_high_res"])
                + torch.sigmoid(out_b["stage2_raw_pred_masks_high_res"])
            )
            out = dict(out_f)
            out["pred_masks_high_res"] = torch.logit(
                native_probability.clamp(1e-4, 1.0 - 1e-4)
            )
            out["stage2_raw_pred_masks_high_res"] = torch.logit(
                raw_probability.clamp(1e-4, 1.0 - 1e-4)
            )
            out["stage2_forward_object_score_logits"] = out_f["multistep_object_score_logits"][-1]
            out["stage2_reverse_object_score_logits"] = out_b["multistep_object_score_logits"][-1]
            fused.append(out)
        return fused


def bidirectional_mixed_outputs(
    core_model,
    batch,
    initial_mask_frames: Sequence[int],
    _previous_hard_tyx: torch.Tensor,
    points: Sequence[CorrectionPoint],
    base_backbone_out: dict | None = None,
    trace: list[dict] | None = None,
) -> list[dict]:
    """Compatibility entry point for official persistent multi-frame replay.

    The hard-volume argument is intentionally ignored.  SAM2 natively carries
    each direction's continuous per-frame logits and same-frame point history.
    """
    state = OfficialMultiFrameBidirectionalState(
        core_model, batch, initial_mask_frames, base_backbone_out, trace
    )
    for point in points:
        state.add_click(point)
    return state.outputs()


def stacked_logits(outputs: Sequence[dict]) -> torch.Tensor:
    """Differentiable raw decoder logits used exclusively by Stage2 loss."""
    return torch.stack(
        [output["stage2_raw_pred_masks_high_res"][:, 0] for output in outputs], dim=0
    )


def stacked_native_logits(outputs: Sequence[dict]) -> torch.Tensor:
    """Native SAM2 output logits used by the interaction state and oracle."""
    return torch.stack([output["pred_masks_high_res"][:, 0] for output in outputs], dim=0)


def hard_prediction(outputs: Sequence[dict]) -> torch.Tensor:
    logits = stacked_native_logits(outputs)
    if logits.shape[1] != 1:
        raise ValueError(f"Stage2 requires patient-level batch size 1, got {tuple(logits.shape)}")
    return logits[:, 0].detach().gt(0.0)
