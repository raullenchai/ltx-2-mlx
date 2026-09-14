"""Paired latent evaluation for compressed ancestral stage-1 transitions."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.distillation_evaluator import _fuse_checkpoint
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.training_strategies.stage1_transition_distill import (
    Stage1TransitionDistillConfig,
    Stage1TransitionDistillStrategy,
)


def read_stage1_checkpoint_metadata(path: str | Path) -> dict[str, str]:
    with safe_open(str(path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
    if metadata.get("distillation") != "stage1_transition":
        raise ValueError("checkpoint is not marked as stage1_transition distillation")
    sigma = float(metadata.get("stage1_sigma", "nan"))
    target_sigma = float(metadata.get("stage1_target_sigma", "nan"))
    if not 0 < sigma <= 1 or not 0 <= target_sigma < sigma:
        raise ValueError("stage-1 checkpoint requires 0 <= target sigma < start sigma <= 1")
    return metadata


def _strategy(metadata: dict[str, str]) -> Stage1TransitionDistillStrategy:
    return Stage1TransitionDistillStrategy(
        Stage1TransitionDistillConfig(
            sigma=float(metadata["stage1_sigma"]),
            target_sigma=float(metadata["stage1_target_sigma"]),
            video_start_latents_dir=metadata["stage1_video_start_latents_dir"],
            video_terminal_latents_dir=metadata["stage1_video_target_latents_dir"],
            audio_start_latents_dir=metadata["stage1_audio_start_latents_dir"],
            audio_terminal_latents_dir=metadata["stage1_audio_target_latents_dir"],
            conditions_dir=metadata.get("stage1_conditions_dir", "stage1_conditions"),
        )
    )


def _add_batch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _add_batch(item) for key, item in value.items()}
    if isinstance(value, mx.array):
        return mx.expand_dims(value, axis=0)
    return value


def load_stage1_student(
    model_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    transformer_file: str | None = None,
) -> tuple[nn.Module, dict[str, str]]:
    metadata = read_stage1_checkpoint_metadata(checkpoint_path)
    model = load_transformer(model_dir, transformer_file=transformer_file)
    _fuse_checkpoint(model, checkpoint_path, metadata)
    return model, metadata


def predict_stage1_transition(model: nn.Module, inputs, sigma: float, target_sigma: float):
    assert inputs.audio is not None
    start = time.perf_counter()
    video_velocity, audio_velocity = model(
        video_latent=inputs.video.latent,
        video_text_embeds=inputs.video.context,
        video_positions=inputs.video.positions,
        timestep=inputs.video.sigma,
        video_timesteps=inputs.video.timesteps,
        audio_latent=inputs.audio.latent,
        audio_text_embeds=inputs.audio.context,
        audio_positions=inputs.audio.positions,
        audio_timesteps=inputs.audio.timesteps,
    )
    delta = sigma - target_sigma
    video = inputs.video.latent - delta * video_velocity
    audio = inputs.audio.latent - delta * audio_velocity
    mx.eval(video, audio)
    return video, audio, time.perf_counter() - start


def _metrics(predicted: mx.array, target: mx.array) -> dict[str, float]:
    predicted = predicted.astype(mx.float32)
    target = target.astype(mx.float32)
    error = predicted - target
    mse = float(mx.mean(mx.square(error)).item())
    target_rms = float(mx.sqrt(mx.mean(mx.square(target))).item())
    return {
        "mse": mse,
        "mae": float(mx.mean(mx.abs(error)).item()),
        "relative_rmse": math.sqrt(mse) / max(target_rms, 1e-12),
    }


def evaluate_stage1_student(
    model: nn.Module,
    data_root: str | Path,
    metadata: dict[str, str],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    strategy = _strategy(metadata)
    dataset = PrecomputedDataset(str(data_root), data_sources=strategy.get_data_sources())
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    if count <= 0:
        raise ValueError("evaluation dataset is empty")
    sigma = strategy.config.sigma
    target_sigma = strategy.config.target_sigma
    samples = []
    for index in range(count):
        inputs = strategy.prepare_training_inputs(_add_batch(dataset[index]), sigma_sampler=None)
        assert inputs.audio is not None
        video, audio, elapsed = predict_stage1_transition(model, inputs, sigma, target_sigma)
        delta = sigma - target_sigma
        video_target = inputs.video.latent - delta * inputs.video_targets
        audio_target = inputs.audio.latent - delta * inputs.audio_targets
        samples.append(
            {
                "index": index,
                "seconds": elapsed,
                "video": _metrics(video, video_target),
                "audio": _metrics(audio, audio_target),
            }
        )
    return {
        "transition": [sigma, target_sigma],
        "samples": samples,
        "mean_seconds": sum(item["seconds"] for item in samples) / count,
        "video_mse": sum(item["video"]["mse"] for item in samples) / count,
        "audio_mse": sum(item["audio"]["mse"] for item in samples) / count,
    }
