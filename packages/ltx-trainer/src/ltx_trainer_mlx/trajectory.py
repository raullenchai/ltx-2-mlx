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

    frames, height, width = spatial_dims
    expected_video_tokens = frames * height * width
    if video_start.ndim != 3 or video_start.shape[1] != expected_video_tokens:
        raise ValueError(f"video trajectory has {video_start.shape[1]} tokens, expected {expected_video_tokens}")

    root = Path(output_root)
    precomputed = root if root.name == ".precomputed" else root / ".precomputed"
    for source in _SOURCES:
        (precomputed / source).mkdir(parents=True, exist_ok=True)

    latent_name = f"latent_{index:04d}.safetensors"
    paths = {
        source: precomputed / source / (f"condition_{index:04d}.safetensors" if source == "conditions" else latent_name)
        for source in _SOURCES
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
