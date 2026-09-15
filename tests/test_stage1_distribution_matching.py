from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_trainer_mlx.stage1_distribution_matching import (
    GeneratedStage1Clean,
    Stage1DmdWeights,
    TeacherStage1Clean,
    _at_sigma,
    balance_decoded_loss,
    decoded_feature_losses,
    fake_score_objective,
    generate_stage1_clean,
    terminal_detail_losses,
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
    assert Stage1DmdWeights(paired=0, distribution_matching=0, terminal=1, detail=1)
    with pytest.raises(ValueError, match="requires a decoded"):
        Stage1DmdWeights(decoded_balance=1)


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


def test_terminal_detail_losses_are_zero_for_exact_teacher_match() -> None:
    video = mx.arange(1 * 8 * 2).reshape(1, 8, 2).astype(mx.float32)
    audio = mx.arange(1 * 4 * 2).reshape(1, 4, 2).astype(mx.float32)
    generated = GeneratedStage1Clean(video=video, audio=audio, paired_loss=mx.array(0.0))
    teacher = TeacherStage1Clean(video=video, audio=audio)

    terminal, detail = terminal_detail_losses(generated, teacher, (2, 2, 2))

    assert float(terminal.item()) == 0.0
    assert float(detail.item()) == 0.0


def test_terminal_detail_losses_detect_spatial_temporal_and_audio_errors() -> None:
    teacher_video = mx.zeros((1, 8, 1))
    student_video = teacher_video.at[:, 7, :].add(2.0)
    teacher_audio = mx.zeros((1, 3, 1))
    student_audio = teacher_audio.at[:, 2, :].add(3.0)
    generated = GeneratedStage1Clean(video=student_video, audio=student_audio, paired_loss=mx.array(0.0))
    teacher = TeacherStage1Clean(video=teacher_video, audio=teacher_audio)

    terminal, detail = terminal_detail_losses(generated, teacher, (2, 2, 2))

    assert float(terminal.item()) > 0.0
    assert float(detail.item()) > 0.0


def test_terminal_detail_losses_reject_bad_grid() -> None:
    generated = GeneratedStage1Clean(
        video=mx.zeros((1, 8, 1)), audio=mx.zeros((1, 3, 1)), paired_loss=mx.array(0.0)
    )
    teacher = TeacherStage1Clean(video=mx.zeros((1, 8, 1)), audio=mx.zeros((1, 3, 1)))

    with pytest.raises(ValueError, match="video dimensions"):
        terminal_detail_losses(generated, teacher, (1, 2, 3))


class IdentityDecoder(nn.Module):
    def decode(self, latent):
        return latent[:, :1]


def test_decoded_feature_losses_backpropagate_rgb_window_error() -> None:
    teacher_video = mx.zeros((1, 8, 2))
    student_video = teacher_video.at[:, 7, 0].add(1.0)
    audio = mx.zeros((1, 3, 1))
    generated = GeneratedStage1Clean(video=student_video, audio=audio, paired_loss=mx.array(0.0))
    teacher = TeacherStage1Clean(video=teacher_video, audio=audio)

    terminal, detail = decoded_feature_losses(
        IdentityDecoder(), generated, teacher, (2, 2, 2), (0, 0, 0, 2, 2, 2)
    )

    assert float(terminal.item()) > 0.0
    assert float(detail.item()) > 0.0


def test_decoded_feature_losses_reject_crop_outside_grid() -> None:
    video = mx.zeros((1, 8, 2))
    audio = mx.zeros((1, 3, 1))
    generated = GeneratedStage1Clean(video=video, audio=audio, paired_loss=mx.array(0.0))
    teacher = TeacherStage1Clean(video=video, audio=audio)

    with pytest.raises(ValueError, match="exceeds"):
        decoded_feature_losses(IdentityDecoder(), generated, teacher, (2, 2, 2), (1, 0, 0, 2, 2, 2))


def test_decoded_loss_balance_matches_paired_value_without_normalizer_gradient() -> None:
    paired = mx.array(0.5)
    decoded = mx.array(0.125)

    balanced, scale = balance_decoded_loss(paired, decoded, ratio=1.0)
    gradient = mx.grad(lambda value: balance_decoded_loss(paired, value, ratio=1.0)[0])(decoded)

    assert mx.allclose(balanced, paired)
    assert float(scale.item()) == 4.0
    assert float(gradient.item()) == 4.0
