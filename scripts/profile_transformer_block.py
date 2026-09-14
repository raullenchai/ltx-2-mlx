#!/usr/bin/env python3
"""Profile one real LTX-2.5 transformer block at production token shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from ltx_pipelines_mlx.utils._orchestration import load_transformer


def _measure(fn, runs: int) -> dict[str, object]:
    warmup = fn()
    mx.eval(warmup)
    del warmup
    mx.clear_cache()

    times = []
    peaks = []
    for _ in range(runs):
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = fn()
        mx.eval(output)
        times.append(time.perf_counter() - started)
        peaks.append(mx.get_peak_memory())
        del output
        mx.clear_cache()
    return {
        "runs_s": times,
        "median_s": statistics.median(times),
        "peak_mlx_bytes": max(peaks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--size", default="768x512")
    parser.add_argument("--text-tokens", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    width, height = (int(part) for part in args.size.lower().split("x", 1))
    latent_frames = (args.frames + 7) // 8
    video_tokens = latent_frames * (height // 32) * (width // 32)
    audio_tokens = round(args.frames / 24 * 25)

    model_dir = Path(args.model)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(args.model))
    model = load_transformer(model_dir / "transformer-distilled.safetensors", low_ram_streaming=True)
    streamer = object.__getattribute__(model, "_streamer")
    block = object.__getattribute__(model, "_shared_block")
    config = model.config
    streamer.bind(block, args.block)
    mx.eval(block.parameters())

    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    video = mx.zeros((1, video_tokens, config.video_dim), dtype=dtype)
    audio = mx.zeros((1, audio_tokens, config.audio_dim), dtype=dtype)
    video_text = mx.zeros((1, args.text_tokens, config.video_dim), dtype=dtype)
    audio_text = mx.zeros((1, args.text_tokens, config.audio_dim), dtype=dtype)
    mx.eval(video, audio, video_text, audio_text)

    operations = {
        "video_self_attention": lambda: block.attn1(video),
        "audio_self_attention": lambda: block.audio_attn1(audio),
        "video_text_cross_attention": lambda: block.attn2(video, encoder_hidden_states=video_text),
        "audio_text_cross_attention": lambda: block.audio_attn2(audio, encoder_hidden_states=audio_text),
        "audio_to_video_attention": lambda: block.audio_to_video_attn(video, encoder_hidden_states=audio),
        "video_to_audio_attention": lambda: block.video_to_audio_attn(audio, encoder_hidden_states=video),
        "video_feed_forward": lambda: block.ff(video),
        "audio_feed_forward": lambda: block.audio_ff(audio),
    }
    results = {name: _measure(fn, args.runs) for name, fn in operations.items()}
    print(
        json.dumps(
            {
                "model": str(model_dir),
                "frames": args.frames,
                "size": [width, height],
                "video_tokens": video_tokens,
                "audio_tokens": audio_tokens,
                "text_tokens": args.text_tokens,
                "block": args.block,
                "dtype": args.dtype,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
