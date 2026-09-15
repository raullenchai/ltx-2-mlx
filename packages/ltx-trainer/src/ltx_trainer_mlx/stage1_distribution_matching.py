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
    """Weights for the paired anchor and distribution-matching field."""

    paired: float = 1.0
    distribution_matching: float = 0.05

    def __post_init__(self) -> None:
        if self.paired < 0 or self.distribution_matching < 0:
            raise ValueError("DMD loss weights must be non-negative")
        if self.paired + self.distribution_matching == 0:
            raise ValueError("at least one DMD loss weight must be positive")


@dataclass(frozen=True)
class Stage1DmdLosses:
    total: mx.array
    paired: mx.array
    distribution_matching: mx.array


@dataclass(frozen=True)
class GeneratedStage1Clean:
    video: mx.array
    audio: mx.array
    paired_loss: mx.array


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
    fake_score: nn.Module,
    strategy: Stage1Transition,
    inputs: ModelInputs,
    batch: dict[str, Any],
    sigma: mx.array | float,
    video_noise: mx.array,
    audio_noise: mx.array,
    weights: Stage1DmdWeights,
) -> Stage1DmdLosses:
    """Return the paired-anchor plus DMD generator objective."""
    generated = generate_stage1_clean(generator, strategy, inputs, batch)
    dm_loss = distribution_matching_loss(
        generator,
        fake_score,
        generated,
        inputs,
        sigma,
        video_noise,
        audio_noise,
    )
    return Stage1DmdLosses(
        total=weights.paired * generated.paired_loss + weights.distribution_matching * dm_loss,
        paired=generated.paired_loss,
        distribution_matching=dm_loss,
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

