"""Paired latent evaluator for one-step stage-2 distillation checkpoints."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from ltx_core_mlx.loader.fuse_loras import apply_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.training_strategies.stage2_terminal_distill import (
    Stage2TerminalDistillConfig,
    Stage2TerminalDistillStrategy,
)


def read_checkpoint_metadata(path: str | Path) -> dict[str, str]:
    """Read and validate metadata required for terminal inference."""
    with safe_open(str(path), framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
    if metadata.get("distillation") != "stage2_terminal":
        raise ValueError("checkpoint is not marked as stage2_terminal distillation")
    if int(metadata.get("stage2_steps", "0")) != 1:
        raise ValueError("terminal checkpoint must declare stage2_steps=1")
    return metadata


def terminal_sigma_schedule(metadata: dict[str, str]) -> list[float]:
    """Return the actual product schedule, including the terminal zero."""
    sigma = float(metadata["stage2_sigma"])
    if not 0 < sigma <= 1:
        raise ValueError("stage2_sigma must be in (0, 1]")
    return [sigma, 0.0]


def _fuse_checkpoint(model: nn.Module, checkpoint_path: str | Path, metadata: dict[str, str]) -> None:
    raw = dict(mx.load(str(checkpoint_path)))
    lora = {key.removeprefix("diffusion_model."): value for key, value in raw.items()}
    rank = int(metadata.get("lora_rank", "0"))
    if rank <= 0:
        a_weights = [value for key, value in lora.items() if key.endswith(".lora_A.weight")]
        if not a_weights:
            raise ValueError("checkpoint contains no LoRA A weights")
        rank = a_weights[0].shape[0]
    alpha = float(metadata.get("lora_alpha", rank))
    strength = alpha / rank

    flat_model = {key: value for key, value in nn.utils.tree_flatten(model.parameters())}
    adapter_prefixes = {key[: -len(".lora_A.weight")] for key in lora if key.endswith(".lora_A.weight")}
    unmatched = sorted(prefix for prefix in adapter_prefixes if f"{prefix}.weight" not in flat_model)
    if unmatched:
        preview = ", ".join(unmatched[:3])
        raise ValueError(f"{len(unmatched)} LoRA target(s) do not match the base transformer: {preview}")
    fused = apply_loras(
        StateDict(sd=flat_model, size=0, dtype=set()),
        [LoraStateDictWithStrength(StateDict(sd=lora, size=0, dtype=set()), strength)],
    )
    model.load_weights(list(fused.sd.items()), strict=False)
    mx.eval(model.parameters())


def load_terminal_student(
    model_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    transformer_file: str | None = None,
) -> tuple[nn.Module, dict[str, str]]:
    """Load the distilled base transformer and fuse a terminal LoRA."""
    metadata = read_checkpoint_metadata(checkpoint_path)
    terminal_sigma_schedule(metadata)
    model = load_transformer(model_dir, transformer_file=transformer_file)
    _fuse_checkpoint(model, checkpoint_path, metadata)
    return model, metadata


def load_terminal_baseline(
    model_dir: str | Path,
    *,
    sigma: float = 0.909375,
    transformer_file: str | None = None,
) -> tuple[nn.Module, dict[str, str]]:
    """Load the unadapted base for a one-evaluation baseline."""
    metadata = {
        "distillation": "stage2_terminal",
        "stage2_sigma": str(sigma),
        "stage2_steps": "1",
    }
    terminal_sigma_schedule(metadata)
    return load_transformer(model_dir, transformer_file=transformer_file), metadata


def _add_batch(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _add_batch(item) for key, item in value.items()}
    if isinstance(value, mx.array):
        return mx.expand_dims(value, axis=0)
    return value


def prepare_trajectory_inputs(sample: dict[str, Any], metadata: dict[str, str]):
    """Convert one unbatched precomputed sample into transformer inputs."""
    sigma = terminal_sigma_schedule(metadata)[0]
    strategy = Stage2TerminalDistillStrategy(Stage2TerminalDistillConfig(sigma=sigma))
    return strategy.prepare_training_inputs(_add_batch(sample), sigma_sampler=None)


def predict_terminal(model: nn.Module, inputs, sigma: float) -> tuple[mx.array, mx.array, float]:
    """Run the student's single terminal evaluation."""
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
    video_pred = inputs.video.latent - sigma * video_velocity
    audio_pred = inputs.audio.latent - sigma * audio_velocity
    mx.eval(video_pred, audio_pred)
    return video_pred, audio_pred, time.perf_counter() - start


def _modality_metrics(predicted: mx.array, target: mx.array) -> dict[str, float]:
    predicted = predicted.astype(mx.float32)
    target = target.astype(mx.float32)
    error = predicted - target
    mse = float(mx.mean(mx.square(error)).item())
    mae = float(mx.mean(mx.abs(error)).item())
    target_rms = float(mx.sqrt(mx.mean(mx.square(target))).item())
    relative_rmse = math.sqrt(mse) / max(target_rms, 1e-12)
    pred_flat = predicted.reshape(-1)
    target_flat = target.reshape(-1)
    cosine = float(
        (mx.sum(pred_flat * target_flat) / (mx.linalg.norm(pred_flat) * mx.linalg.norm(target_flat) + 1e-12)).item()
    )
    return {"mse": mse, "mae": mae, "relative_rmse": relative_rmse, "cosine": cosine}


def evaluate_terminal_student(
    model: nn.Module,
    data_root: str | Path,
    metadata: dict[str, str],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compare one student evaluation with paired teacher terminal latents."""
    sigma = terminal_sigma_schedule(metadata)[0]
    strategy = Stage2TerminalDistillStrategy(Stage2TerminalDistillConfig(sigma=sigma))
    dataset = PrecomputedDataset(str(data_root), data_sources=strategy.get_data_sources())
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    if count <= 0:
        raise ValueError("evaluation dataset is empty")

    samples: list[dict[str, Any]] = []
    for index in range(count):
        inputs = prepare_trajectory_inputs(dataset[index], metadata)
        assert inputs.audio is not None and inputs.audio_targets is not None
        video_pred, audio_pred, elapsed = predict_terminal(model, inputs, sigma)
        video_target = inputs.video.latent - sigma * inputs.video_targets
        audio_target = inputs.audio.latent - sigma * inputs.audio_targets
        samples.append(
            {
                "index": index,
                "seconds": elapsed,
                "video": _modality_metrics(video_pred, video_target),
                "audio": _modality_metrics(audio_pred, audio_target),
            }
        )

    aggregate: dict[str, Any] = {"samples": count, "mean_seconds": sum(x["seconds"] for x in samples) / count}
    for modality in ("video", "audio"):
        aggregate[modality] = {
            metric: sum(sample[modality][metric] for sample in samples) / count
            for metric in ("mse", "mae", "relative_rmse", "cosine")
        }
    return {"schedule": [sigma, 0.0], "aggregate": aggregate, "per_sample": samples}
