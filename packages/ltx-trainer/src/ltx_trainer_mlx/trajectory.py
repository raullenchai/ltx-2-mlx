"""Persistence helpers for stage-2 teacher trajectories."""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier

_SOURCES = (
    "stage2_video_start_latents",
    "stage2_video_terminal_latents",
    "stage2_audio_start_latents",
    "stage2_audio_terminal_latents",
    "conditions",
)
_INTERMEDIATE_SOURCES = (
    "stage2_video_intermediate_latents",
    "stage2_audio_intermediate_latents",
)


def save_stage1_trajectory_step(
    output_root: str | Path,
    index: int,
    step_index: int,
    *,
    sigma: float,
    video: mx.array,
    audio: mx.array,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    spatial_dims: tuple[int, int, int],
    frame_rate: float,
    noise_seed: int,
    seed: int,
    prompt: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically persist one boundary of an ancestral stage-1 trajectory."""
    if index < 0 or step_index < 0:
        raise ValueError("trajectory and step indices must be non-negative")
    if not 0 <= sigma <= 1:
        raise ValueError("stage-1 sigma must be in [0, 1]")
    frames, height, width = spatial_dims
    expected_video_tokens = frames * height * width
    if video.ndim != 3 or video.shape[0] != 1 or video.shape[1] != expected_video_tokens:
        raise ValueError(f"stage-1 video trajectory shape {video.shape} does not match {spatial_dims}")
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"stage-1 audio trajectory must have batch size 1, got {audio.shape}")

    root = Path(output_root)
    precomputed = root if root.name == ".precomputed" else root / ".precomputed"
    video_source = f"stage1_video_step_{step_index:02d}"
    audio_source = f"stage1_audio_step_{step_index:02d}"
    sources = [video_source, audio_source]
    if step_index == 0:
        sources.append("stage1_conditions")
    for source in sources:
        (precomputed / source).mkdir(parents=True, exist_ok=True)

    latent_name = f"latent_{index:04d}.safetensors"
    paths = {
        video_source: precomputed / video_source / latent_name,
        audio_source: precomputed / audio_source / latent_name,
    }
    if step_index == 0:
        paths["stage1_conditions"] = precomputed / "stage1_conditions" / latent_name
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"stage-1 trajectory {index} step {step_index} has existing components")

    metadata_arrays = {
        "num_frames": mx.array([frames], dtype=mx.int32),
        "height": mx.array([height], dtype=mx.int32),
        "width": mx.array([width], dtype=mx.int32),
        "fps": mx.array([frame_rate], dtype=mx.float32),
        "sigma": mx.array([sigma], dtype=mx.float32),
        "seed": mx.array([seed], dtype=mx.int32),
        "noise_seed": mx.array([noise_seed], dtype=mx.int64),
        "step_index": mx.array([step_index], dtype=mx.int32),
    }
    audio_unpatched = AudioPatchifier().unpatchify(audio)
    tensors = {
        video_source: {"latents": _as_bfloat16(_without_batch(video, "video")), **metadata_arrays},
        audio_source: {"latents": _as_bfloat16(_without_batch(audio_unpatched, "audio")), **metadata_arrays},
    }
    if step_index == 0:
        tensors["stage1_conditions"] = {
            "video_prompt_embeds": _as_bfloat16(_without_batch(video_text_embeds, "video_text_embeds")),
            "audio_prompt_embeds": _as_bfloat16(_without_batch(audio_text_embeds, "audio_text_embeds")),
            "prompt_attention_mask": mx.ones((video_text_embeds.shape[1],), dtype=mx.float32),
        }

    temporary: dict[str, Path] = {}
    try:
        for source, path in paths.items():
            temporary[source] = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
            mx.save_safetensors(
                str(temporary[source]),
                tensors[source],
                {"prompt": prompt} if source == video_source else None,
            )
        for source, path in paths.items():
            temporary[source].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    return paths


def _without_batch(array: mx.array, name: str) -> mx.array:
    if array.ndim < 1 or array.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1, got shape {array.shape}")
    return array[0]


def _as_bfloat16(array: mx.array) -> mx.array:
    return array.astype(mx.bfloat16)


def save_stage2_trajectory(
    output_root: str | Path,
    index: int,
    *,
    video_start: mx.array,
    video_terminal: mx.array,
    audio_start: mx.array,
    audio_terminal: mx.array,
    video_intermediate: mx.array | None = None,
    audio_intermediate: mx.array | None = None,
    intermediate_sigma: float | None = None,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    spatial_dims: tuple[int, int, int],
    frame_rate: float,
    sigma: float,
    seed: int,
    prompt: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically save one trajectory in ``PrecomputedDataset`` layout."""
    if index < 0:
        raise ValueError("trajectory index must be non-negative")
    if video_start.shape != video_terminal.shape:
        raise ValueError("video start and terminal shapes must match")
    if audio_start.shape != audio_terminal.shape:
        raise ValueError("audio start and terminal shapes must match")
    intermediate_values = (video_intermediate, audio_intermediate, intermediate_sigma)
    if any(value is not None for value in intermediate_values) and not all(
        value is not None for value in intermediate_values
    ):
        raise ValueError("video, audio, and sigma intermediate values must be provided together")
    if video_intermediate is not None and video_intermediate.shape != video_start.shape:
        raise ValueError("video start and intermediate shapes must match")
    if audio_intermediate is not None and audio_intermediate.shape != audio_start.shape:
        raise ValueError("audio start and intermediate shapes must match")
    if intermediate_sigma is not None and not 0 < intermediate_sigma < sigma:
        raise ValueError("intermediate_sigma must be in (0, sigma)")

    frames, height, width = spatial_dims
    expected_video_tokens = frames * height * width
    if video_start.ndim != 3 or video_start.shape[1] != expected_video_tokens:
        raise ValueError(f"video trajectory has {video_start.shape[1]} tokens, expected {expected_video_tokens}")

    root = Path(output_root)
    precomputed = root if root.name == ".precomputed" else root / ".precomputed"
    sources = _SOURCES + (_INTERMEDIATE_SOURCES if video_intermediate is not None else ())
    for source in sources:
        (precomputed / source).mkdir(parents=True, exist_ok=True)

    latent_name = f"latent_{index:04d}.safetensors"
    paths = {
        source: precomputed / source / (f"condition_{index:04d}.safetensors" if source == "conditions" else latent_name)
        for source in sources
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"trajectory {index} already has {len(existing)} saved component(s)")

    audio_patchifier = AudioPatchifier()
    audio_start_unpatched = audio_patchifier.unpatchify(audio_start)
    audio_terminal_unpatched = audio_patchifier.unpatchify(audio_terminal)
    metadata_arrays = {
        "num_frames": mx.array([frames], dtype=mx.int32),
        "height": mx.array([height], dtype=mx.int32),
        "width": mx.array([width], dtype=mx.int32),
        "fps": mx.array([frame_rate], dtype=mx.float32),
        "sigma": mx.array([sigma], dtype=mx.float32),
        "seed": mx.array([seed], dtype=mx.int32),
    }
    tensors = {
        "stage2_video_start_latents": {
            "latents": _as_bfloat16(_without_batch(video_start, "video_start")),
            **metadata_arrays,
        },
        "stage2_video_terminal_latents": {
            "latents": _as_bfloat16(_without_batch(video_terminal, "video_terminal")),
            **metadata_arrays,
        },
        "stage2_audio_start_latents": {
            "latents": _as_bfloat16(_without_batch(audio_start_unpatched, "audio_start")),
        },
        "stage2_audio_terminal_latents": {
            "latents": _as_bfloat16(_without_batch(audio_terminal_unpatched, "audio_terminal")),
        },
        "conditions": {
            "video_prompt_embeds": _as_bfloat16(_without_batch(video_text_embeds, "video_text_embeds")),
            "audio_prompt_embeds": _as_bfloat16(_without_batch(audio_text_embeds, "audio_text_embeds")),
            "prompt_attention_mask": mx.ones((video_text_embeds.shape[1],), dtype=mx.float32),
        },
    }
    if video_intermediate is not None and audio_intermediate is not None and intermediate_sigma is not None:
        intermediate_metadata = {**metadata_arrays, "target_sigma": mx.array([intermediate_sigma], dtype=mx.float32)}
        tensors["stage2_video_intermediate_latents"] = {
            "latents": _as_bfloat16(_without_batch(video_intermediate, "video_intermediate")),
            **intermediate_metadata,
        }
        audio_intermediate_unpatched = audio_patchifier.unpatchify(audio_intermediate)
        tensors["stage2_audio_intermediate_latents"] = {
            "latents": _as_bfloat16(_without_batch(audio_intermediate_unpatched, "audio_intermediate")),
        }

    temporary: dict[str, Path] = {}
    try:
        for source, path in paths.items():
            temporary[source] = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
            mx.save_safetensors(
                str(temporary[source]),
                tensors[source],
                {"prompt": prompt} if source == "stage2_video_start_latents" else None,
            )
        for source, path in paths.items():
            temporary[source].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)

    return paths
