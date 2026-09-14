#!/usr/bin/env python3
"""Measure whether Q8 LTX-2.5 LoRA backward fits on a 48 GiB Mac."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_video_positions,
)
from ltx_pipelines_mlx.utils._orchestration import load_transformer
from ltx_trainer_mlx.trainer import LtxvTrainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video-tokens", type=int, default=144)
    parser.add_argument("--audio-tokens", type=int, default=26)
    parser.add_argument("--text-tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()

    mx.random.seed(42)
    mx.reset_peak_memory()
    started = time.perf_counter()
    model = load_transformer(
        Path(args.model) / "transformer-distilled.safetensors", low_ram_streaming=False
    )
    load_seconds = time.perf_counter() - started

    trainer = object.__new__(LtxvTrainer)
    trainer._transformer = model
    trainer._config = SimpleNamespace(
        lora=SimpleNamespace(
            rank=args.rank,
            alpha=args.rank,
            target_modules=["to_q", "to_k", "to_v"],
        )
    )
    trainer._setup_lora()
    model.gradient_checkpointing = True

    video = mx.random.normal((1, args.video_tokens, 128), dtype=mx.bfloat16)
    audio = mx.random.normal((1, args.audio_tokens, 128), dtype=mx.bfloat16)
    video_text = mx.random.normal(
        (1, args.text_tokens, model.config.video_dim), dtype=mx.bfloat16
    )
    audio_text = mx.random.normal(
        (1, args.text_tokens, model.config.audio_dim), dtype=mx.bfloat16
    )
    sigma = mx.array([0.909375], dtype=mx.bfloat16)
    video_timesteps = mx.full((1, args.video_tokens), sigma.item(), dtype=mx.bfloat16)
    audio_timesteps = mx.full((1, args.audio_tokens), sigma.item(), dtype=mx.bfloat16)

    # The exact grid is irrelevant to memory feasibility; use a 1-D spatial strip.
    video_positions = compute_video_positions(1, 1, args.video_tokens, frame_rate=24.0)
    audio_positions = compute_audio_positions(args.audio_tokens)
    target_video = mx.zeros_like(video)
    target_audio = mx.zeros_like(audio)
    mx.eval(video, audio, video_text, audio_text, target_video, target_audio)

    def loss_fn(_unused):
        video_pred, audio_pred = model(
            video_latent=video,
            audio_latent=audio,
            timestep=sigma,
            video_text_embeds=video_text,
            audio_text_embeds=audio_text,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        return mx.mean(mx.square(video_pred - target_video)) + mx.mean(
            mx.square(audio_pred - target_audio)
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    mx.reset_peak_memory()
    step_started = time.perf_counter()
    loss, grads = loss_and_grad(None)
    mx.eval(loss, grads)
    step_seconds = time.perf_counter() - step_started
    trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters())
    )

    print(
        json.dumps(
            {
                "load_seconds": load_seconds,
                "step_seconds": step_seconds,
                "loss": float(loss.item()),
                "active_gib": mx.get_active_memory() / 2**30,
                "peak_gib": mx.get_peak_memory() / 2**30,
                "trainable_parameters": trainable,
                "video_tokens": args.video_tokens,
                "audio_tokens": args.audio_tokens,
                "text_tokens": args.text_tokens,
                "rank": args.rank,
                "pid": os.getpid(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
