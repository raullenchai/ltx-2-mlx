import mlx.core as mx
import pytest

from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import ancestral_euler_step, ancestral_step_noise
from ltx_trainer_mlx.training_strategies.base_strategy import ModalityInputs, ModelInputs
from scripts.materialize_stage1_teacher_correction import _strategy, teacher_correct_transition


class _ZeroVelocityModel:
    def __call__(self, *, video_latent, audio_latent, **kwargs):
        return mx.zeros_like(video_latent), mx.zeros_like(audio_latent)


def _modality(latent: mx.array) -> ModalityInputs:
    tokens = latent.shape[1]
    return ModalityInputs(
        enabled=True,
        latent=latent,
        sigma=mx.array([DISTILLED_SIGMAS[3]]),
        timesteps=mx.full((1, tokens), DISTILLED_SIGMAS[3]),
        positions=mx.zeros((1, tokens, 3)),
        context=mx.zeros((1, 2, 4)),
    )


def test_teacher_correction_replays_every_original_noise_lane() -> None:
    video = mx.ones((1, 3, 2), dtype=mx.bfloat16)
    audio = mx.ones((1, 2, 2), dtype=mx.bfloat16)
    inputs = ModelInputs(
        video=_modality(video),
        audio=_modality(audio),
        video_targets=mx.zeros_like(video),
        audio_targets=mx.zeros_like(audio),
        video_loss_mask=mx.ones((1, 3), dtype=mx.bool_),
        audio_loss_mask=mx.ones((1, 2), dtype=mx.bool_),
    )

    predicted_video, predicted_audio, _ = teacher_correct_transition(
        _ZeroVelocityModel(),
        inputs,
        noise_seed=10042,
        start_step=3,
        target_step=5,
    )

    expected_video, expected_audio = video, audio
    for index in range(3, 5):
        video_noise, audio_noise = ancestral_step_noise(
            10042,
            len(DISTILLED_SIGMAS) - 1,
            index,
            expected_video.shape,
            expected_audio.shape,
        )
        expected_video = ancestral_euler_step(
            expected_video,
            expected_video,
            DISTILLED_SIGMAS[index],
            DISTILLED_SIGMAS[index + 1],
            video_noise,
        ).astype(mx.bfloat16)
        expected_audio = ancestral_euler_step(
            expected_audio,
            expected_audio,
            DISTILLED_SIGMAS[index],
            DISTILLED_SIGMAS[index + 1],
            audio_noise,
        ).astype(mx.bfloat16)

    assert mx.array_equal(predicted_video, expected_video).item()
    assert mx.array_equal(predicted_audio, expected_audio).item()


def test_teacher_correction_rejects_invalid_span() -> None:
    with pytest.raises(ValueError, match="start < end"):
        _strategy(5, 5)
