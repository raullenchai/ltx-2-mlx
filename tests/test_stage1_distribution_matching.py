from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_trainer_mlx.stage1_distribution_matching import (
    Stage1DmdWeights,
    _at_sigma,
    fake_score_objective,
    generate_stage1_clean,
)
from ltx_trainer_mlx.training_strategies.base_strategy import ModalityInputs, ModelInputs


class ScaleModel(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.weight = mx.array(scale)

    def __call__(self, *, video_latent, audio_latent, **kwargs):
        del kwargs
        return self.weight * video_latent, self.weight * audio_latent


class Transition:
    config = SimpleNamespace(sigma=0.8, target_sigma=0.4)

    def advance_transition(self, video_velocity, audio_velocity, inputs, batch):
        del batch
        return (
            inputs.video.latent - 0.4 * video_velocity,
            inputs.audio.latent - 0.4 * audio_velocity,
        )

    def compute_loss(self, video_pred, audio_pred, inputs):
        return mx.mean(mx.square(video_pred - inputs.video_targets)) + mx.mean(
            mx.square(audio_pred - inputs.audio_targets)
        )


def _modality(latent: mx.array, sigma: float) -> ModalityInputs:
    batch, tokens, _ = latent.shape
    return ModalityInputs(
        enabled=True,
        latent=latent,
        sigma=mx.full((batch,), sigma),
        timesteps=mx.full((batch, tokens), sigma),
        positions=mx.zeros((1, tokens, 1)),
        context=mx.zeros((batch, 1, 2)),
    )


def _inputs() -> ModelInputs:
    video = mx.ones((1, 2, 2))
    audio = mx.full((1, 3, 2), 2.0)
    return ModelInputs(
        video=_modality(video, 0.8),
        audio=_modality(audio, 0.8),
        video_targets=mx.zeros_like(video),
        audio_targets=mx.zeros_like(audio),
        video_loss_mask=mx.ones((1, 2), dtype=mx.bool_),
        audio_loss_mask=mx.ones((1, 3), dtype=mx.bool_),
    )


def test_dmd_weights_reject_empty_or_negative_objective() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Stage1DmdWeights(paired=0, distribution_matching=0)
    with pytest.raises(ValueError, match="non-negative"):
        Stage1DmdWeights(paired=-1)


def test_at_sigma_rebuilds_scalar_and_per_token_timesteps() -> None:
    original = _inputs().video
    latent = mx.zeros_like(original.latent)

    updated = _at_sigma(original, latent, mx.array([0.2]))

    assert updated.latent is latent
    assert mx.allclose(updated.sigma, mx.array([0.2]))
    assert mx.allclose(updated.timesteps, mx.full((1, 2), 0.2))


def test_at_sigma_rejects_wrong_batch_shape() -> None:
    with pytest.raises(ValueError, match="one value per batch"):
        _at_sigma(_inputs().video, mx.zeros((1, 2, 2)), mx.array([0.1, 0.2]))


def test_generate_stage1_clean_completes_intermediate_transition() -> None:
    generated = generate_stage1_clean(ScaleModel(0.5), Transition(), _inputs(), {})

    # x7 = x3 - (0.8 - 0.4) * 0.5*x3 = 0.8*x3, then
    # x0 = x7 - 0.4 * 0.5*x7 = 0.64*x3.
    assert mx.allclose(generated.video, mx.full((1, 2, 2), 0.64))
    assert mx.allclose(generated.audio, mx.full((1, 3, 2), 1.28))


def test_generate_stage1_clean_rejects_terminal_transition() -> None:
    strategy = Transition()
    strategy.config = SimpleNamespace(sigma=0.4, target_sigma=0.0)
    with pytest.raises(ValueError, match="above sigma zero"):
        generate_stage1_clean(ScaleModel(0.5), strategy, _inputs(), {})


def test_fake_score_objective_uses_detached_generated_clean() -> None:
    generated = generate_stage1_clean(ScaleModel(0.5), Transition(), _inputs(), {})
    loss = fake_score_objective(
        ScaleModel(0.0),
        generated,
        _inputs(),
        0.25,
        mx.zeros_like(generated.video),
        mx.zeros_like(generated.audio),
    )

    assert bool(mx.isfinite(loss).item())
    assert float(loss.item()) > 0
