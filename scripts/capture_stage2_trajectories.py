#!/usr/bin/env python3
"""Capture deterministic stage-2 teacher trajectories without decoding video."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_trainer_mlx.trajectory import save_stage2_trajectory


def _read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


@dataclass(frozen=True)
class CaptureRequest:
    index: int
    prompt: str
    width: int
    height: int
    frames: int
    frame_rate: float
    seed: int
    bucket: str | None = None


def _read_manifest(path: Path) -> list[CaptureRequest]:
    """Read and validate a JSONL capture manifest."""
    requests: list[CaptureRequest] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
            request = CaptureRequest(
                index=int(item["index"]),
                prompt=str(item["prompt"]).strip(),
                width=int(item["width"]),
                height=int(item["height"]),
                frames=int(item["frames"]),
                frame_rate=float(item.get("frame_rate", 24.0)),
                seed=int(item["seed"]),
                bucket=str(item["bucket"]) if "bucket" in item else None,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest item on line {line_number}: {error}") from error
        if request.index < 0 or not request.prompt:
            raise ValueError(f"invalid index or empty prompt on manifest line {line_number}")
        if request.width % 32 or request.height % 32 or request.frames % 8 != 1:
            raise ValueError(f"invalid LTX dimensions on manifest line {line_number}")
        requests.append(request)
    if not requests:
        raise ValueError(f"no capture requests found in {path}")
    indices = [request.index for request in requests]
    if len(indices) != len(set(indices)):
        raise ValueError("manifest trajectory indices must be unique")
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path, help="One prompt per non-empty line.")
    source.add_argument("--manifest", type=Path, help="JSONL requests with fixed dimensions, seed, and index.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bucket", help="Capture only this manifest bucket while retaining global indices.")
    parser.add_argument("--low-ram-streaming", action="store_true")
    args = parser.parse_args()

    if args.manifest is not None:
        requests = _read_manifest(args.manifest)
        if args.bucket is not None:
            requests = [request for request in requests if request.bucket == args.bucket]
    else:
        if args.bucket is not None:
            parser.error("--bucket requires --manifest")
        requests = [
            CaptureRequest(
                index=index,
                prompt=prompt,
                width=args.width,
                height=args.height,
                frames=args.frames,
                frame_rate=args.frame_rate,
                seed=args.seed + index,
            )
            for index, prompt in enumerate(_read_prompts(args.prompts))
        ]
    if args.limit is not None:
        requests = requests[: args.limit]
    if not requests:
        raise ValueError("no capture requests remain after filtering")

    for progress, request in enumerate(requests, start=1):
        # A low-memory generation releases the VAE encoder and upsampler.
        # Construct per sample so a multi-prompt capture never reuses that
        # intentionally freed component state. Existing complete samples are
        # skipped so interrupted captures can resume safely.
        precomputed = args.output / ".precomputed"
        latent_name = f"latent_{request.index:04d}.safetensors"
        expected = [
            precomputed / source / latent_name
            for source in (
                "stage2_video_start_latents",
                "stage2_video_terminal_latents",
                "stage2_audio_start_latents",
                "stage2_audio_terminal_latents",
                "stage2_video_intermediate_latents",
                "stage2_audio_intermediate_latents",
            )
        ]
        expected.append(precomputed / "conditions" / f"condition_{request.index:04d}.safetensors")
        present = [path for path in expected if path.exists()]
        if len(present) == len(expected):
            print(f"skipped existing trajectory {progress}/{len(requests)} (index {request.index})")
            continue
        if present:
            raise RuntimeError(
                f"trajectory {request.index} is incomplete ({len(present)}/{len(expected)} files); inspect it"
            )
        pipeline = DistilledPipeline(
            args.model,
            low_memory=True,
            low_ram_streaming=args.low_ram_streaming,
        )
        callback = partial(save_stage2_trajectory, args.output, request.index)
        pipeline.generate_two_stage(
            request.prompt,
            height=request.height,
            width=request.width,
            num_frames=request.frames,
            frame_rate=request.frame_rate,
            seed=request.seed,
            stage2_trajectory_callback=callback,
        )
        print(f"captured trajectory {progress}/{len(requests)} (index {request.index})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
