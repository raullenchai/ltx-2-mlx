#!/usr/bin/env python3
"""Capture deterministic stage-2 teacher trajectories without decoding video."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_trainer_mlx.trajectory import save_stage2_trajectory


def _read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="One prompt per non-empty line.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--low-ram-streaming", action="store_true")
    args = parser.parse_args()

    prompts = _read_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    for index, prompt in enumerate(prompts):
        # A low-memory generation releases the VAE encoder and upsampler.
        # Construct per sample so a multi-prompt capture never reuses that
        # intentionally freed component state. Existing complete samples are
        # skipped so interrupted captures can resume safely.
        precomputed = args.output / ".precomputed"
        latent_name = f"latent_{index:04d}.safetensors"
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
        expected.append(precomputed / "conditions" / f"condition_{index:04d}.safetensors")
        present = [path for path in expected if path.exists()]
        if len(present) == len(expected):
            print(f"skipped existing trajectory {index + 1}/{len(prompts)}")
            continue
        if present:
            raise RuntimeError(f"trajectory {index} is incomplete ({len(present)}/{len(expected)} files); inspect it")
        pipeline = DistilledPipeline(
            args.model,
            low_memory=True,
            low_ram_streaming=args.low_ram_streaming,
        )
        callback = partial(save_stage2_trajectory, args.output, index)
        pipeline.generate_two_stage(
            prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.frame_rate,
            seed=args.seed + index,
            stage2_trajectory_callback=callback,
        )
        print(f"captured trajectory {index + 1}/{len(prompts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
