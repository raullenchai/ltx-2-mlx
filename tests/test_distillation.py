from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import ancestral_euler_step, ancestral_span_noise, ancestral_step_noise
from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.distillation import (
    ancestral_velocity_target,
    euler_step,
    lora_disabled,
    terminal_velocity_target,
    transition_velocity_target,
)
from ltx_trainer_mlx.distillation_evaluator import terminal_sigma_schedule
from ltx_trainer_mlx.training_strategies.stage1_transition_distill import (
    Stage1TransitionDistillConfig,
    Stage1TransitionDistillStrategy,
)
from ltx_trainer_mlx.training_strategies.stage2_terminal_distill import (
    Stage2TerminalDistillConfig,
    Stage2TerminalDistillStrategy,
)
from ltx_trainer_mlx.trajectory import save_stage1_trajectory_step, save_stage2_trajectory


class _Adapter(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.lora_a = mx.zeros((1, 1))
        self.lora_b = mx.zeros((1, 1))
        self.scale = scale


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = _Adapter(2.0)
        self.nested = [_Adapter(3.0)]
        self.plain = nn.Linear(1, 1)


def test_terminal_velocity_target_reaches_teacher_terminal() -> None:
    sample = mx.array([[[2.0, -1.0], [0.5, 4.0]]])
    terminal = mx.array([[[0.25, 0.5], [-2.0, 1.0]]])
    sigma = mx.array(0.909375)

    velocity = terminal_velocity_target(sample, terminal, sigma)
    reconstructed = euler_step(sample, velocity, sigma, 0.0)

    assert mx.allclose(reconstructed, terminal, atol=1e-6).item()


def test_terminal_checkpoint_schedule_jumps_to_zero() -> None:
    assert terminal_sigma_schedule({"stage2_sigma": "0.909375"}) == [0.909375, 0.0]


def test_progressive_checkpoint_schedule_keeps_teacher_final_step() -> None:
    assert terminal_sigma_schedule({"stage2_sigma": "0.909375", "stage2_target_sigma": "0.421875"}) == [
        0.909375,
        0.421875,
        0.0,
    ]


def test_transition_velocity_target_reaches_teacher_intermediate() -> None:
    sample = mx.array([[[2.0, -1.0]]])
    target = mx.array([[[0.25, 0.5]]])
    sigma, target_sigma = 0.909375, 0.421875

    velocity = transition_velocity_target(sample, target, sigma, target_sigma)
    reconstructed = euler_step(sample, velocity, sigma, target_sigma)

    assert mx.allclose(reconstructed, target, atol=1e-6).item()


def test_ancestral_velocity_target_reaches_teacher_intermediate() -> None:
    sample = mx.array([[[2.0, -1.0]]])
    target = mx.array([[[0.25, 0.5]]])
    noise = mx.array([[[0.3, -0.2]]])
    sigma, target_sigma = 0.975, 0.725

    velocity = ancestral_velocity_target(sample, target, noise, sigma, target_sigma)
    denoised = sample - sigma * velocity
    reconstructed = ancestral_euler_step(sample, denoised, sigma, target_sigma, noise)

    assert mx.allclose(reconstructed, target, atol=1e-6).item()


def test_ancestral_step_noise_is_stable_by_original_step() -> None:
    first = ancestral_step_noise(10042, 8, 4, (1, 3, 2), (1, 2, 2))
    second = ancestral_step_noise(10042, 8, 4, (1, 3, 2), (1, 2, 2))
    other = ancestral_step_noise(10042, 8, 5, (1, 3, 2), (1, 2, 2))

    assert mx.array_equal(first[0], second[0]).item()
    assert mx.array_equal(first[1], second[1]).item()
    assert not mx.array_equal(first[0], other[0]).item()


def test_ancestral_span_noise_preserves_fine_step_stochastic_forcing() -> None:
    seed = 10042
    shape = (1, 3, 2)
    fine = mx.zeros(shape)
    deterministic = mx.zeros(shape)
    for index in range(3):
        noise, _ = ancestral_step_noise(seed, 8, index, shape, shape)
        sigma, sigma_next = DISTILLED_SIGMAS[index : index + 2]
        fine = ancestral_euler_step(fine, mx.zeros_like(fine), sigma, sigma_next, noise)
        deterministic = ancestral_euler_step(
            deterministic, mx.zeros_like(deterministic), sigma, sigma_next, mx.zeros_like(noise)
        )

    span, _ = ancestral_span_noise(seed, 8, 0, 3, DISTILLED_SIGMAS, shape, shape)
    coarse = ancestral_euler_step(
        mx.zeros(shape), mx.zeros(shape), DISTILLED_SIGMAS[0], DISTILLED_SIGMAS[3], span
    )
    coarse_deterministic = ancestral_euler_step(
        mx.zeros(shape),
        mx.zeros(shape),
        DISTILLED_SIGMAS[0],
        DISTILLED_SIGMAS[3],
        mx.zeros(shape),
    )

    assert mx.allclose(fine - deterministic, coarse - coarse_deterministic, atol=1e-6).item()


def test_ancestral_span_noise_rejects_terminal_span() -> None:
    with pytest.raises(ValueError, match="terminal spans"):
        ancestral_span_noise(42, 8, 7, 8, DISTILLED_SIGMAS, (1, 2, 2), (1, 2, 2))


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_terminal_velocity_target_rejects_non_positive_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        terminal_velocity_target(mx.zeros((1,)), mx.zeros((1,)), sigma)


def test_terminal_velocity_target_rejects_per_sample_sigma() -> None:
    with pytest.raises(ValueError, match="one shared sigma"):
        terminal_velocity_target(mx.zeros((2, 1)), mx.zeros((2, 1)), mx.array([0.9, 0.8]))


def test_lora_disabled_restores_scales_after_exception() -> None:
    model = _Model()

    with pytest.raises(RuntimeError, match="teacher failed"), lora_disabled(model):
        assert model.first.scale == 0.0
        assert model.nested[0].scale == 0.0
        raise RuntimeError("teacher failed")

    assert model.first.scale == 2.0
    assert model.nested[0].scale == 3.0


def _trajectory_batch() -> dict:
    video_start = mx.arange(128 * 2 * 2 * 2).reshape(1, 128, 2, 2, 2).astype(mx.float32) / 100
    video_terminal = video_start * 0.75
    audio_start = mx.arange(8 * 3 * 16).reshape(1, 8, 3, 16).astype(mx.float32) / 100
    audio_terminal = audio_start * 0.5
    return {
        "video_start": {
            "latents": video_start,
            "num_frames": mx.array([2]),
            "height": mx.array([2]),
            "width": mx.array([2]),
            "fps": mx.array([24.0]),
        },
        "video_terminal": {"latents": video_terminal},
        "audio_start": {"latents": audio_start},
        "audio_terminal": {"latents": audio_terminal},
        "conditions": {
            "video_prompt_embeds": mx.zeros((1, 4, 4096)),
            "audio_prompt_embeds": mx.zeros((1, 4, 2048)),
            "prompt_attention_mask": mx.ones((1, 4)),
        },
    }


def test_stage2_strategy_builds_exact_terminal_targets() -> None:
    strategy = Stage2TerminalDistillStrategy(Stage2TerminalDistillConfig())
    batch = _trajectory_batch()
    inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)

    reconstructed_video = euler_step(inputs.video.latent, inputs.video_targets, inputs.video.sigma[0], 0.0)
    reconstructed_audio = euler_step(inputs.audio.latent, inputs.audio_targets, inputs.audio.sigma[0], 0.0)
    expected_video, _ = strategy._video_patchifier.patchify(batch["video_terminal"]["latents"])
    expected_audio, _ = strategy._audio_patchifier.patchify(batch["audio_terminal"]["latents"])

    assert mx.allclose(reconstructed_video, expected_video, atol=1e-6).item()
    assert mx.allclose(reconstructed_audio, expected_audio, atol=1e-6).item()
    assert float(strategy.compute_loss(inputs.video_targets, inputs.audio_targets, inputs).item()) == 0.0


def test_stage2_strategy_declares_all_trajectory_sources() -> None:
    strategy = Stage2TerminalDistillStrategy(Stage2TerminalDistillConfig())

    assert strategy.requires_audio
    assert set(strategy.get_data_sources().values()) == {
        "video_start",
        "video_terminal",
        "audio_start",
        "audio_terminal",
        "conditions",
    }
    assert strategy.get_checkpoint_metadata() == {
        "distillation": "stage2_terminal",
        "stage2_sigma": 0.909375,
        "stage2_target_sigma": 0.0,
        "stage2_steps": 1,
        "stage2_video_target_latents_dir": "stage2_video_terminal_latents",
        "stage2_audio_target_latents_dir": "stage2_audio_terminal_latents",
    }


def test_stage2_progressive_strategy_targets_intermediate() -> None:
    strategy = Stage2TerminalDistillStrategy(
        Stage2TerminalDistillConfig(
            target_sigma=0.421875,
            video_terminal_latents_dir="stage2_video_intermediate_latents",
            audio_terminal_latents_dir="stage2_audio_intermediate_latents",
        )
    )
    batch = _trajectory_batch()
    batch["video_terminal"]["target_sigma"] = mx.array([[0.421875]])
    inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)

    reconstructed = euler_step(inputs.video.latent, inputs.video_targets, 0.909375, 0.421875)
    expected, _ = strategy._video_patchifier.patchify(batch["video_terminal"]["latents"])
    assert mx.allclose(reconstructed, expected, atol=1e-6).item()
    assert strategy.get_checkpoint_metadata()["stage2_steps"] == 2


def test_stage2_strategy_rejects_trajectory_sigma_mismatch() -> None:
    strategy = Stage2TerminalDistillStrategy(Stage2TerminalDistillConfig())
    batch = _trajectory_batch()
    batch["video_start"]["sigma"] = mx.array([[0.725]])

    with pytest.raises(ValueError, match="does not match configured sigma"):
        strategy.prepare_training_inputs(batch, sigma_sampler=None)


def test_stage2_trajectory_round_trips_through_precomputed_dataset(tmp_path) -> None:
    video_start = mx.arange(2 * 2 * 2 * 128).reshape(1, 8, 128).astype(mx.float32)
    video_terminal = video_start * 0.9
    audio_start = mx.arange(3 * 128).reshape(1, 3, 128).astype(mx.float32)
    audio_terminal = audio_start * 0.8
    video_text = mx.zeros((1, 4, 4096))
    audio_text = mx.zeros((1, 4, 2048))

    paths = save_stage2_trajectory(
        tmp_path,
        0,
        video_start=video_start,
        video_terminal=video_terminal,
        audio_start=audio_start,
        audio_terminal=audio_terminal,
        video_text_embeds=video_text,
        audio_text_embeds=audio_text,
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        sigma=0.909375,
        seed=42,
        prompt="test prompt",
    )
    strategy = Stage2TerminalDistillStrategy(Stage2TerminalDistillConfig())
    dataset = PrecomputedDataset(str(tmp_path), data_sources=strategy.get_data_sources())
    sample = dataset[0]

    assert len(paths) == 5
    assert len(dataset) == 1
    assert sample["video_start"]["latents"].shape == (128, 2, 2, 2)
    assert sample["audio_start"]["latents"].shape == (8, 3, 16)
    assert sample["video_start"]["latents"].dtype == mx.bfloat16
    assert sample["conditions"]["video_prompt_embeds"].dtype == mx.bfloat16
    assert mx.array_equal(
        sample["video_start"]["latents"].reshape(128, -1).T,
        video_start[0].astype(mx.bfloat16),
    ).item()


def test_stage1_trajectory_step_round_trips_through_precomputed_dataset(tmp_path) -> None:
    video = mx.arange(8 * 128).reshape(1, 8, 128).astype(mx.float32)
    audio = mx.arange(3 * 128).reshape(1, 3, 128).astype(mx.float32)
    paths = save_stage1_trajectory_step(
        tmp_path,
        3,
        0,
        sigma=1.0,
        video=video,
        audio=audio,
        video_text_embeds=mx.zeros((1, 4, 4096)),
        audio_text_embeds=mx.zeros((1, 4, 2048)),
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        noise_seed=10042,
        seed=42,
        prompt="test prompt",
    )
    dataset = PrecomputedDataset(
        str(tmp_path),
        data_sources={
            "stage1_video_step_00": "video",
            "stage1_audio_step_00": "audio",
            "stage1_conditions": "conditions",
        },
    )
    sample = dataset[0]

    assert len(paths) == 3
    assert sample["video"]["latents"].shape == (8, 128)
    assert sample["audio"]["latents"].shape == (8, 3, 16)
    assert int(sample["video"]["noise_seed"].item()) == 10042
    assert sample["video"]["latents"].dtype == mx.bfloat16


def test_stage1_trajectory_can_reuse_external_conditions(tmp_path) -> None:
    paths = save_stage1_trajectory_step(
        tmp_path,
        3,
        0,
        sigma=1.0,
        video=mx.zeros((1, 8, 128)),
        audio=mx.zeros((1, 3, 128)),
        video_text_embeds=mx.zeros((1, 4, 4096)),
        audio_text_embeds=mx.zeros((1, 4, 2048)),
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        noise_seed=10042,
        seed=42,
        prompt="test prompt",
        save_conditions=False,
    )

    assert len(paths) == 2
    assert not (tmp_path / ".precomputed/stage1_conditions").exists()


def test_stage1_transition_strategy_reconstructs_captured_target(tmp_path) -> None:
    video_start = mx.arange(8 * 128).reshape(1, 8, 128).astype(mx.float32) / 100
    video_target = video_start * 0.8
    audio_start = mx.arange(3 * 128).reshape(1, 3, 128).astype(mx.float32) / 100
    audio_target = audio_start * 0.7
    common = dict(
        output_root=tmp_path,
        index=0,
        video_text_embeds=mx.zeros((1, 4, 4096)),
        audio_text_embeds=mx.zeros((1, 4, 2048)),
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        noise_seed=10042,
        seed=42,
        prompt="test",
    )
    save_stage1_trajectory_step(step_index=0, sigma=1.0, video=video_start, audio=audio_start, **common)
    save_stage1_trajectory_step(step_index=2, sigma=0.9875, video=video_target, audio=audio_target, **common)
    config = Stage1TransitionDistillConfig(
        sigma=1.0,
        target_sigma=0.9875,
        video_start_latents_dir="stage1_video_step_00",
        video_terminal_latents_dir="stage1_video_step_02",
        audio_start_latents_dir="stage1_audio_step_00",
        audio_terminal_latents_dir="stage1_audio_step_02",
    )
    strategy = Stage1TransitionDistillStrategy(config)
    dataset = PrecomputedDataset(str(tmp_path), data_sources=strategy.get_data_sources())
    batch = {
        key: ({name: mx.expand_dims(value, 0) for name, value in item.items()} if isinstance(item, dict) else item)
        for key, item in dataset[0].items()
    }
    inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
    assert inputs.audio is not None and inputs.audio_targets is not None

    reconstructed_video = euler_step(inputs.video.latent, inputs.video_targets, 1.0, 0.9875)
    reconstructed_audio = euler_step(inputs.audio.latent, inputs.audio_targets, 1.0, 0.9875)
    assert mx.allclose(reconstructed_video, video_target.astype(mx.bfloat16), atol=1e-4).item()
    assert mx.allclose(reconstructed_audio, audio_target.astype(mx.bfloat16), atol=1e-4).item()


def test_stage1_noise_coupled_strategy_reconstructs_captured_target(tmp_path) -> None:
    video_start = mx.arange(8 * 128).reshape(1, 8, 128).astype(mx.float32) / 100
    video_target = video_start * 0.8
    audio_start = mx.arange(3 * 128).reshape(1, 3, 128).astype(mx.float32) / 100
    audio_target = audio_start * 0.7
    common = dict(
        output_root=tmp_path,
        index=0,
        video_text_embeds=mx.zeros((1, 4, 4096)),
        audio_text_embeds=mx.zeros((1, 4, 2048)),
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        noise_seed=10042,
        seed=42,
        prompt="test",
    )
    save_stage1_trajectory_step(step_index=0, sigma=1.0, video=video_start, audio=audio_start, **common)
    save_stage1_trajectory_step(step_index=2, sigma=0.9875, video=video_target, audio=audio_target, **common)
    strategy = Stage1TransitionDistillStrategy(
        Stage1TransitionDistillConfig(
            sigma=1.0,
            target_sigma=0.9875,
            video_start_latents_dir="stage1_video_step_00",
            video_terminal_latents_dir="stage1_video_step_02",
            audio_start_latents_dir="stage1_audio_step_00",
            audio_terminal_latents_dir="stage1_audio_step_02",
            ancestral_noise_step_index=0,
        )
    )
    dataset = PrecomputedDataset(str(tmp_path), data_sources=strategy.get_data_sources())
    batch = {
        key: ({name: mx.expand_dims(value, 0) for name, value in item.items()} if isinstance(item, dict) else item)
        for key, item in dataset[0].items()
    }

    inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
    assert inputs.audio is not None and inputs.audio_targets is not None
    reconstructed_video, reconstructed_audio = strategy.advance_transition(
        inputs.video_targets, inputs.audio_targets, inputs, batch
    )
    assert mx.allclose(reconstructed_video, video_target.astype(mx.bfloat16), atol=1e-4).item()
    assert mx.allclose(reconstructed_audio, audio_target.astype(mx.bfloat16), atol=1e-4).item()
    assert strategy.get_checkpoint_metadata()["stage1_sampler"] == "ancestral"


def test_stage1_span_noise_strategy_reconstructs_captured_target(tmp_path) -> None:
    video_start = mx.arange(8 * 128).reshape(1, 8, 128).astype(mx.float32) / 100
    video_target = video_start * 0.8
    audio_start = mx.arange(3 * 128).reshape(1, 3, 128).astype(mx.float32) / 100
    audio_target = audio_start * 0.7
    common = dict(
        output_root=tmp_path,
        index=0,
        video_text_embeds=mx.zeros((1, 4, 4096)),
        audio_text_embeds=mx.zeros((1, 4, 2048)),
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        noise_seed=10042,
        seed=42,
        prompt="test",
    )
    save_stage1_trajectory_step(step_index=0, sigma=1.0, video=video_start, audio=audio_start, **common)
    save_stage1_trajectory_step(step_index=3, sigma=0.98125, video=video_target, audio=audio_target, **common)
    strategy = Stage1TransitionDistillStrategy(
        Stage1TransitionDistillConfig(
            sigma=1.0,
            target_sigma=0.98125,
            video_start_latents_dir="stage1_video_step_00",
            video_terminal_latents_dir="stage1_video_step_03",
            audio_start_latents_dir="stage1_audio_step_00",
            audio_terminal_latents_dir="stage1_audio_step_03",
            ancestral_noise_step_index=0,
            ancestral_noise_step_end_index=3,
            ancestral_noise_reference_sigmas=DISTILLED_SIGMAS,
        )
    )
    dataset = PrecomputedDataset(str(tmp_path), data_sources=strategy.get_data_sources())
    batch = {
        key: ({name: mx.expand_dims(value, 0) for name, value in item.items()} if isinstance(item, dict) else item)
        for key, item in dataset[0].items()
    }

    inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
    assert inputs.audio is not None and inputs.audio_targets is not None
    reconstructed_video, reconstructed_audio = strategy.advance_transition(
        inputs.video_targets, inputs.audio_targets, inputs, batch
    )
    assert mx.allclose(reconstructed_video, video_target.astype(mx.bfloat16), atol=1e-4).item()
    assert mx.allclose(reconstructed_audio, audio_target.astype(mx.bfloat16), atol=1e-4).item()
    metadata = strategy.get_checkpoint_metadata()
    assert metadata["stage1_sampler"] == "ancestral_span_v2"
    assert metadata["stage1_noise_step_end_index"] == 3


def test_stage1_noise_coupling_rejects_terminal_transition() -> None:
    with pytest.raises(ValueError, match="remain above sigma zero"):
        Stage1TransitionDistillConfig(
            sigma=0.725,
            target_sigma=0.0,
            video_start_latents_dir="video-start",
            video_terminal_latents_dir="video-target",
            audio_start_latents_dir="audio-start",
            audio_terminal_latents_dir="audio-target",
            ancestral_noise_step_index=6,
        )


def test_stage2_trajectory_saves_optional_intermediate(tmp_path) -> None:
    video = mx.arange(8 * 128).reshape(1, 8, 128).astype(mx.float32)
    audio = mx.arange(3 * 128).reshape(1, 3, 128).astype(mx.float32)
    paths = save_stage2_trajectory(
        tmp_path,
        0,
        video_start=video,
        video_terminal=video * 0.5,
        audio_start=audio,
        audio_terminal=audio * 0.5,
        video_intermediate=video * 0.75,
        audio_intermediate=audio * 0.75,
        intermediate_sigma=0.421875,
        video_text_embeds=mx.zeros((1, 4, 4096)),
        audio_text_embeds=mx.zeros((1, 4, 2048)),
        spatial_dims=(2, 2, 2),
        frame_rate=24.0,
        sigma=0.909375,
        seed=42,
        prompt="test",
    )
    strategy = Stage2TerminalDistillStrategy(
        Stage2TerminalDistillConfig(
            target_sigma=0.421875,
            video_terminal_latents_dir="stage2_video_intermediate_latents",
            audio_terminal_latents_dir="stage2_audio_intermediate_latents",
        )
    )
    sample = PrecomputedDataset(str(tmp_path), data_sources=strategy.get_data_sources())[0]

    assert len(paths) == 7
    assert float(sample["video_terminal"]["target_sigma"].item()) == pytest.approx(0.421875)


def test_stage2_trajectory_refuses_partial_overwrite(tmp_path) -> None:
    kwargs = {
        "video_start": mx.zeros((1, 1, 128)),
        "video_terminal": mx.zeros((1, 1, 128)),
        "audio_start": mx.zeros((1, 1, 128)),
        "audio_terminal": mx.zeros((1, 1, 128)),
        "video_text_embeds": mx.zeros((1, 1, 4096)),
        "audio_text_embeds": mx.zeros((1, 1, 2048)),
        "spatial_dims": (1, 1, 1),
        "frame_rate": 24.0,
        "sigma": 0.909375,
        "seed": 42,
        "prompt": "test",
    }
    save_stage2_trajectory(tmp_path, 0, **kwargs)

    with pytest.raises(FileExistsError, match="already has 5"):
        save_stage2_trajectory(tmp_path, 0, **kwargs)
