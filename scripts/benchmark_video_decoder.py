#!/usr/bin/env python3
"""Benchmark the LTX video VAE decoder at production latent shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.utils
from huggingface_hub import snapshot_download

from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--size", default="768x512")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--weight-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--latent-dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    width, height = (int(part) for part in args.size.lower().split("x", 1))
    if args.frames < 9 or args.frames % 8 != 1:
        parser.error("--frames must be 8n+1 and at least 9")
    if width % 32 or height % 32:
        parser.error("--size dimensions must be divisible by 32")

    mx.set_cache_limit(0)
    model_dir = Path(args.model)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(args.model))

    decoder = VideoDecoder()
    weights = load_split_safetensors(model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
    decoder.load_weights(list(weights.items()))
    weight_dtype = mx.bfloat16 if args.weight_dtype == "bf16" else mx.float16
    latent_dtype = mx.bfloat16 if args.latent_dtype == "bf16" else mx.float16
    if args.weight_dtype == "fp16":
        decoder.update(mlx.utils.tree_map(lambda value: value.astype(weight_dtype), decoder.parameters()))

    load_started = time.perf_counter()
    mx.eval(decoder.parameters())
    load_s = time.perf_counter() - load_started

    latent_frames = (args.frames + 7) // 8
    latent = mx.zeros(
        (1, 128, latent_frames, height // 32, width // 32),
        dtype=latent_dtype,
    )
    decode = decoder.decode
    if args.compile:
        decode = mx.compile(decode, inputs=decoder)

    times = []
    for _ in range(args.runs):
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = decode(latent)
        mx.eval(output)
        times.append(time.perf_counter() - started)

    report = {
        "model": args.model,
        "frames": args.frames,
        "size": [width, height],
        "latent_shape": list(latent.shape),
        "output_shape": list(output.shape),
        "weight_dtype": args.weight_dtype,
        "latent_dtype": args.latent_dtype,
        "output_dtype": str(output.dtype),
        "compiled": args.compile,
        "weight_materialize_s": load_s,
        "runs_s": times,
        "median_s": statistics.median(times),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "output_mean": float(mx.mean(output.astype(mx.float32)).item()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
