#!/usr/bin/env python3
"""Capture full ancestral stage-1 boundary trajectories without decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

from capture_stage2_trajectories import CaptureRequest, _read_manifest, _read_prompts

from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_trainer_mlx.trajectory import save_stage1_trajectory_step


class _CaptureCompleteError(Exception):
    """Internal control flow used to stop before upscaling and stage 2."""


def _expected_paths(output: Path, index: int) -> list[Path]:
    precomputed = output / ".precomputed"
    latent = f"latent_{index:04d}.safetensors"
    paths = []
    for step_index in range(len(DISTILLED_SIGMAS)):
        paths.extend(
            [
                precomputed / f"stage1_video_step_{step_index:02d}" / latent,
                precomputed / f"stage1_audio_step_{step_index:02d}" / latent,
            ]
        )
    paths.append(precomputed / "stage1_conditions" / latent)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bucket")
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
        expected = _expected_paths(args.output, request.index)
        present = [path for path in expected if path.exists()]
        if len(present) == len(expected):
            print(f"skipped existing trajectory {progress}/{len(requests)} (index {request.index})")
            continue
        if present:
            raise RuntimeError(
                f"stage-1 trajectory {request.index} is incomplete ({len(present)}/{len(expected)} files); inspect it"
            )

        pipeline = DistilledPipeline(args.model, low_memory=True, low_ram_streaming=args.low_ram_streaming)
        step_index = 0

        def capture(*, _request_index=request.index, **values) -> None:
            nonlocal step_index
            expected_sigma = DISTILLED_SIGMAS[step_index]
            if abs(values["sigma"] - expected_sigma) > 1e-9:
                raise RuntimeError(f"stage-1 sigma {values['sigma']} does not match schedule {expected_sigma}")
            save_stage1_trajectory_step(args.output, _request_index, step_index, **values)
            step_index += 1
            if step_index == len(DISTILLED_SIGMAS):
                raise _CaptureCompleteError

        try:
            pipeline.generate_two_stage(
                request.prompt,
                height=request.height,
                width=request.width,
                num_frames=request.frames,
                frame_rate=request.frame_rate,
                seed=request.seed,
                stage1_trajectory_callback=capture,
            )
        except _CaptureCompleteError:
            pass
        finally:
            del pipeline
            aggressive_cleanup()
        if step_index != len(DISTILLED_SIGMAS):
            raise RuntimeError(f"captured {step_index}/{len(DISTILLED_SIGMAS)} stage-1 boundaries")
        print(f"captured stage-1 trajectory {progress}/{len(requests)} (index {request.index})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
