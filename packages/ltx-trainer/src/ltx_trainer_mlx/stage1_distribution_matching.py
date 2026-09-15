"""Experimental Stage-1 distribution-matching objective for LTX-2.5.

The objective is deliberately separate from :mod:`trainer`: it needs two
optimizers and a second score-model adapter, while ordinary training owns one
model and one optimizer.  Keeping the model-independent math and the LTX
forward orchestration here makes the gradient boundaries reviewable before a
long product-grid run is launched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn

from ltx_trainer_mlx.distillation import lora_disabled
from ltx_trainer_mlx.distribution_matching import (
    distribution_matching_surrogate_loss,
    fake_score_flow_loss,
    normalized_distribution_matching_gradient,
    rectified_flow_noising,
    rectified_flow_x0,
)
from ltx_trainer_mlx.training_strategies.base_strategy import ModalityInputs, ModelInputs


class Stage1Transition(Protocol):
    """Subset of the captured-transition strategy needed by DMD."""

    config: Any

    def advance_transition(
        self,
        video_velocity: mx.array,
        audio_velocity: mx.array,
        inputs: ModelInputs,
        batch: dict[str, Any],
    ) -> tuple[mx.array, mx.array]: ...

    def compute_loss(
        self,
        video_pred: mx.array,
        audio_pred: mx.array | None,
        inputs: ModelInputs,
    ) -> mx.array: ...


@dataclass(frozen=True)
class Stage1DmdWeights:
    """Weights for paired, terminal, detail, and distribution objectives."""

    paired: float = 1.0
    distribution_matching: float = 0.05
    terminal: float = 0.0
    detail: float = 0.0
    decoded_terminal: float = 0.0
    decoded_detail: float = 0.0
    decoded_balance: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.paired,
            self.distribution_matching,
            self.terminal,
            self.detail,
            self.decoded_terminal,
            self.decoded_detail,
        )
        if any(value < 0 for value in (*values, self.decoded_balance)):
            raise ValueError("DMD loss weights must be non-negative")
        if sum(values) == 0:
            raise ValueError("at least one DMD loss weight must be positive")
        if self.decoded_balance and self.decoded_terminal + self.decoded_detail == 0:
            raise ValueError("decoded balance requires a decoded endpoint loss")


@dataclass(frozen=True)
class Stage1DmdLosses:
    total: mx.array
    paired: mx.array
    distribution_matching: mx.array
    terminal: mx.array
    detail: mx.array
    decoded_terminal: mx.array
    decoded_detail: mx.array
    decoded_scale: mx.array


@dataclass(frozen=True)
class GeneratedStage1Clean:
    video: mx.array
    audio: mx.array
    paired_loss: mx.array


@dataclass(frozen=True)
class TeacherStage1Clean:
    """Frozen teacher endpoint aligned to the captured transition sample."""

    video: mx.array
    audio: mx.array


def _at_sigma(modality: ModalityInputs, latent: mx.array, sigma: mx.array | float) -> ModalityInputs:
    sigma_array = mx.array(sigma)
    if sigma_array.size == 1:
        sigma_array = mx.full((latent.shape[0],), float(sigma_array.item()), dtype=latent.dtype)
    if sigma_array.shape != (latent.shape[0],):
        raise ValueError("sigma must be scalar or contain one value per batch item")
    return replace(
        modality,
        latent=latent,
        sigma=sigma_array,
        timesteps=mx.broadcast_to(sigma_array[:, None], latent.shape[:2]),
    )


def _forward(model: nn.Module, video: ModalityInputs, audio: ModalityInputs) -> tuple[mx.array, mx.array]:
    """Run the joint LTX velocity model for a prepared modality pair."""
    result = model(
        video_latent=video.latent,
        audio_latent=audio.latent,
        timestep=video.sigma,
        video_text_embeds=video.context,
        audio_text_embeds=audio.context,
        video_positions=video.positions,
        audio_positions=audio.positions,
        video_attention_mask=video.attention_mask,
        audio_attention_mask=audio.attention_mask,
        video_timesteps=video.timesteps,
        audio_timesteps=audio.timesteps,
    )
    if not isinstance(result, tuple) or len(result) != 2 or result[1] is None:
        raise ValueError("Stage-1 DMD requires joint video/audio velocity predictions")
    return result


def generate_stage1_clean(
    generator: nn.Module,
    strategy: Stage1Transition,
    inputs: ModelInputs,
    batch: dict[str, Any],
) -> GeneratedStage1Clean:
    """Run the learned transition followed by an exact base-model terminal step.

    The terminal forward remains differentiable with respect to its input, but
    its LoRA scale is zero.  Consequently the generator adapter learns through
    the complete Stage-1 output instead of treating the intermediate x7 latent
    as if it were clean data.
    """
    if inputs.audio is None:
        raise ValueError("Stage-1 DMD requires audio inputs")
    target_sigma = float(strategy.config.target_sigma)
    if not 0 < target_sigma < float(strategy.config.sigma):
        raise ValueError("Stage-1 DMD requires an intermediate target above sigma zero")

    video_velocity, audio_velocity = _forward(generator, inputs.video, inputs.audio)
    paired_loss = strategy.compute_loss(video_velocity, audio_velocity, inputs)
    video_target, audio_target = strategy.advance_transition(video_velocity, audio_velocity, inputs, batch)
    terminal_video = _at_sigma(inputs.video, video_target, target_sigma)
    terminal_audio = _at_sigma(inputs.audio, audio_target, target_sigma)
    with lora_disabled(generator):
        terminal_video_velocity, terminal_audio_velocity = _forward(generator, terminal_video, terminal_audio)
    return GeneratedStage1Clean(
        video=rectified_flow_x0(video_target, terminal_video_velocity, target_sigma),
        audio=rectified_flow_x0(audio_target, terminal_audio_velocity, target_sigma),
        paired_loss=paired_loss,
    )


def generate_teacher_stage1_clean(
    teacher: nn.Module,
    strategy: Stage1Transition,
    inputs: ModelInputs,
    batch: dict[str, Any],
) -> TeacherStage1Clean:
    """Complete the captured teacher transition with the exact base terminal step."""
    if inputs.audio is None or inputs.audio_targets is None:
        raise ValueError("Stage-1 teacher completion requires audio targets")
    target_sigma = float(strategy.config.target_sigma)
    if not 0 < target_sigma < float(strategy.config.sigma):
        raise ValueError("Stage-1 teacher completion requires an intermediate target")

    video_target, audio_target = strategy.advance_transition(
        inputs.video_targets,
        inputs.audio_targets,
        inputs,
        batch,
    )
    terminal_video = _at_sigma(inputs.video, video_target, target_sigma)
    terminal_audio = _at_sigma(inputs.audio, audio_target, target_sigma)
    with lora_disabled(teacher):
        video_velocity, audio_velocity = _forward(teacher, terminal_video, terminal_audio)
    return TeacherStage1Clean(
        video=mx.stop_gradient(rectified_flow_x0(video_target, video_velocity, target_sigma)),
        audio=mx.stop_gradient(rectified_flow_x0(audio_target, audio_velocity, target_sigma)),
    )


def _mean_squared_difference(student: mx.array, teacher: mx.array) -> mx.array:
    return mx.mean(mx.square(student - teacher))


def _gradient_residual(student: mx.array, teacher: mx.array, axis: int) -> mx.array:
    """Compare first differences, emphasizing structure rather than latent DC level."""
    if student.shape != teacher.shape:
        raise ValueError("student and teacher detail tensors must have matching shapes")
    if student.shape[axis] < 2:
        return mx.array(0.0, dtype=student.dtype)
    student_gradient = mx.diff(student, axis=axis)
    teacher_gradient = mx.diff(teacher, axis=axis)
    return _mean_squared_difference(student_gradient, teacher_gradient)


def terminal_detail_losses(
    generated: GeneratedStage1Clean,
    teacher: TeacherStage1Clean,
    video_dims: tuple[int, int, int],
) -> tuple[mx.array, mx.array]:
    """Return endpoint reconstruction and spatiotemporal-gradient losses.

    Video tokens are laid out in ``(frames, height, width)`` order.  Comparing
    first differences along every grid axis makes motion trails and lost
    spatial edges expensive even when their contribution to global latent MSE
    is small.  Audio tokens receive the analogous temporal constraint.
    """
    frames, height, width = video_dims
    expected_tokens = frames * height * width
    if generated.video.shape != teacher.video.shape or generated.video.shape[1] != expected_tokens:
        raise ValueError("video dimensions do not match generated teacher endpoints")
    if generated.audio.shape != teacher.audio.shape:
        raise ValueError("audio generated and teacher endpoints must match")

    terminal = _mean_squared_difference(generated.video, teacher.video) + _mean_squared_difference(
        generated.audio, teacher.audio
    )
    batch_size, _, channels = generated.video.shape
    student_video = generated.video.reshape(batch_size, frames, height, width, channels)
    teacher_video = teacher.video.reshape(batch_size, frames, height, width, channels)
    detail = sum(
        (_gradient_residual(student_video, teacher_video, axis) for axis in (1, 2, 3)),
        start=mx.array(0.0, dtype=generated.video.dtype),
    )
    detail = detail + _gradient_residual(generated.audio, teacher.audio, axis=1)
    return terminal, detail


def decoded_feature_losses(
    decoder: nn.Module,
    generated: GeneratedStage1Clean,
    teacher: TeacherStage1Clean,
    video_dims: tuple[int, int, int],
    crop: tuple[int, int, int, int, int, int],
) -> tuple[mx.array, mx.array]:
    """Compare a sparse decoded RGB window while keeping inference unchanged.

    ``crop`` is ``(frame_start, top, left, frames, height, width)`` in latent
    coordinates. The frozen decoder remains differentiable with respect to
    its input, so RGB-domain errors reach only the generator LoRA.
    """
    total_frames, total_height, total_width = video_dims
    frame_start, top, left, frames, height, width = crop
    if min(frame_start, top, left) < 0 or min(frames, height, width) <= 0:
        raise ValueError("decoded crop coordinates and dimensions are invalid")
    if (
        frame_start + frames > total_frames
        or top + height > total_height
        or left + width > total_width
    ):
        raise ValueError("decoded crop exceeds the video latent grid")

    batch_size, video_tokens, channels = generated.video.shape
    if teacher.video.shape != generated.video.shape or video_tokens != total_frames * total_height * total_width:
        raise ValueError("decoded crop video dimensions do not match endpoint tokens")

    def crop_tokens(tokens: mx.array) -> mx.array:
        unpatchified = tokens.reshape(
            batch_size, total_frames, total_height, total_width, channels
        ).transpose(0, 4, 1, 2, 3)
        return unpatchified[
            :,
            :,
            frame_start : frame_start + frames,
            top : top + height,
            left : left + width,
        ]

    student_pixels = decoder.decode(crop_tokens(generated.video)).astype(mx.float32)
    teacher_pixels = mx.stop_gradient(decoder.decode(crop_tokens(teacher.video)).astype(mx.float32))
    terminal = _mean_squared_difference(student_pixels, teacher_pixels)
    detail = sum(
        (_gradient_residual(student_pixels, teacher_pixels, axis) for axis in (2, 3, 4)),
        start=mx.array(0.0, dtype=student_pixels.dtype),
    )
    return terminal, detail


def balance_decoded_loss(
    paired_loss: mx.array,
    decoded_loss: mx.array,
    ratio: float,
) -> tuple[mx.array, mx.array]:
    """Scale decoded loss to a stable fraction of the paired anchor.

    The scale is detached so the generator cannot lower the objective by
    manipulating its normalizer. A zero ratio preserves the unbalanced loss.
    """
    if ratio < 0:
        raise ValueError("decoded balance ratio must be non-negative")
    if ratio == 0:
        return decoded_loss, mx.array(1.0, dtype=decoded_loss.dtype)
    denominator = mx.maximum(decoded_loss, mx.array(1e-8, dtype=decoded_loss.dtype))
    scale = mx.stop_gradient(ratio * paired_loss / denominator)
    return scale * decoded_loss, scale


def _video_dims(batch: dict[str, Any], video_tokens: int) -> tuple[int, int, int]:
    try:
        source = batch["video_start"]
        dims = tuple(int(source[key][0].item()) for key in ("num_frames", "height", "width"))
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Stage-1 endpoint losses require video_start grid metadata") from exc
    if len(dims) != 3 or dims[0] * dims[1] * dims[2] != video_tokens:
        raise ValueError("video_start grid metadata does not match token count")
    return dims


def distribution_matching_loss(
    real_score: nn.Module,
    fake_score: nn.Module,
    generated: GeneratedStage1Clean,
    inputs: ModelInputs,
    sigma: mx.array | float,
    video_noise: mx.array,
    audio_noise: mx.array,
) -> mx.array:
    """Construct the detached DMD field on the complete generated output."""
    if inputs.audio is None:
        raise ValueError("Stage-1 DMD requires audio inputs")
    noisy_video = rectified_flow_noising(generated.video, video_noise, sigma)
    noisy_audio = rectified_flow_noising(generated.audio, audio_noise, sigma)
    score_video = _at_sigma(inputs.video, noisy_video, sigma)
    score_audio = _at_sigma(inputs.audio, noisy_audio, sigma)

    with lora_disabled(real_score):
        real_video_velocity, real_audio_velocity = _forward(real_score, score_video, score_audio)
    fake_video_velocity, fake_audio_velocity = _forward(fake_score, score_video, score_audio)
    real_video_x0 = rectified_flow_x0(noisy_video, real_video_velocity, sigma)
    real_audio_x0 = rectified_flow_x0(noisy_audio, real_audio_velocity, sigma)
    fake_video_x0 = rectified_flow_x0(noisy_video, fake_video_velocity, sigma)
    fake_audio_x0 = rectified_flow_x0(noisy_audio, fake_audio_velocity, sigma)

    video_gradient = normalized_distribution_matching_gradient(generated.video, real_video_x0, fake_video_x0)
    audio_gradient = normalized_distribution_matching_gradient(generated.audio, real_audio_x0, fake_audio_x0)
    return distribution_matching_surrogate_loss(
        generated.video, video_gradient
    ) + distribution_matching_surrogate_loss(generated.audio, audio_gradient)


def generator_objective(
    generator: nn.Module,
    fake_score: nn.Module | None,
    strategy: Stage1Transition,
    inputs: ModelInputs,
    batch: dict[str, Any],
    sigma: mx.array | float,
    video_noise: mx.array,
    audio_noise: mx.array,
    weights: Stage1DmdWeights,
    decoder: nn.Module | None = None,
    decode_crop: tuple[int, int, int, int, int, int] | None = None,
) -> Stage1DmdLosses:
    """Return the paired-anchor plus DMD generator objective."""
    generated = generate_stage1_clean(generator, strategy, inputs, batch)
    zero = mx.array(0.0, dtype=generated.video.dtype)
    if weights.distribution_matching:
        if fake_score is None:
            raise ValueError("distribution matching requires a fake score model")
        dm_loss = distribution_matching_loss(
            generator,
            fake_score,
            generated,
            inputs,
            sigma,
            video_noise,
            audio_noise,
        )
    else:
        dm_loss = zero
    needs_teacher = weights.terminal or weights.detail or weights.decoded_terminal or weights.decoded_detail
    if needs_teacher:
        teacher = generate_teacher_stage1_clean(generator, strategy, inputs, batch)
        video_dims = _video_dims(batch, generated.video.shape[1])
        terminal_loss, detail_loss = terminal_detail_losses(
            generated,
            teacher,
            video_dims,
        )
    else:
        terminal_loss, detail_loss = zero, zero
        teacher = None
        video_dims = None
    if weights.decoded_terminal or weights.decoded_detail:
        if decoder is None or decode_crop is None or teacher is None or video_dims is None:
            raise ValueError("decoded endpoint losses require a decoder and latent crop")
        decoded_terminal_loss, decoded_detail_loss = decoded_feature_losses(
            decoder,
            generated,
            teacher,
            video_dims,
            decode_crop,
        )
    else:
        decoded_terminal_loss, decoded_detail_loss = zero, zero
    decoded_loss = (
        weights.decoded_terminal * decoded_terminal_loss
        + weights.decoded_detail * decoded_detail_loss
    )
    decoded_loss, decoded_scale = balance_decoded_loss(
        generated.paired_loss,
        decoded_loss,
        weights.decoded_balance,
    )
    return Stage1DmdLosses(
        total=(
            weights.paired * generated.paired_loss
            + weights.distribution_matching * dm_loss
            + weights.terminal * terminal_loss
            + weights.detail * detail_loss
            + decoded_loss
        ),
        paired=generated.paired_loss,
        distribution_matching=dm_loss,
        terminal=terminal_loss,
        detail=detail_loss,
        decoded_terminal=decoded_terminal_loss,
        decoded_detail=decoded_detail_loss,
        decoded_scale=decoded_scale,
    )


def fake_score_objective(
    fake_score: nn.Module,
    generated: GeneratedStage1Clean,
    inputs: ModelInputs,
    sigma: mx.array | float,
    video_noise: mx.array,
    audio_noise: mx.array,
) -> mx.array:
    """Train the fake score on a detached complete Stage-1 generator sample."""
    if inputs.audio is None:
        raise ValueError("Stage-1 DMD requires audio inputs")
    video_clean = mx.stop_gradient(generated.video)
    audio_clean = mx.stop_gradient(generated.audio)
    noisy_video = rectified_flow_noising(video_clean, video_noise, sigma)
    noisy_audio = rectified_flow_noising(audio_clean, audio_noise, sigma)
    video_inputs = _at_sigma(inputs.video, noisy_video, sigma)
    audio_inputs = _at_sigma(inputs.audio, noisy_audio, sigma)
    video_velocity, audio_velocity = _forward(fake_score, video_inputs, audio_inputs)
    return fake_score_flow_loss(video_velocity, video_clean, video_noise) + fake_score_flow_loss(
        audio_velocity, audio_clean, audio_noise
    )
