#!/usr/bin/env python3
"""Capture full ancestral stage-1 boundary trajectories without decoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open

try:
    from scripts.capture_stage2_trajectories import CaptureRequest, _read_manifest, _read_prompts
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
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


def _reuse_condition(source_root: Path, output: Path, index: int) -> Path:
    """Hard-link an existing Stage-2 condition under the Stage-1 filename."""
    source_precomputed = source_root / ".precomputed" if (source_root / ".precomputed").is_dir() else source_root
    source = source_precomputed / "conditions" / f"condition_{index:04d}.safetensors"
    if not source.is_file():
        raise FileNotFoundError(f"reused condition not found: {source}")
    with safe_open(source, framework="numpy") as condition:
        required = {"video_prompt_embeds", "audio_prompt_embeds", "prompt_attention_mask"}
        missing = required.difference(condition.keys())
    if missing:
        raise ValueError(f"reused condition {source} is missing tensors: {sorted(missing)}")
    destination = output / ".precomputed" / "stage1_conditions" / f"latent_{index:04d}.safetensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise FileExistsError(f"reused condition destination has different content: {destination}")
    else:
        os.link(source, destination)
    return destination


def _reuse_manifest_prompts(source_root: Path) -> dict[int, str]:
    root = source_root.parent if source_root.name == ".precomputed" else source_root
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"condition reuse requires its source manifest: {manifest}")
    prompts: dict[int, str] = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        index = record.get("index")
        prompt = record.get("prompt")
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(prompt, str) or not prompt:
            raise ValueError(f"invalid condition source manifest record at line {line_number}")
        if index in prompts:
            raise ValueError(f"duplicate condition source manifest index: {index}")
        prompts[index] = prompt
    return prompts


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
    parser.add_argument(
        "--reuse-conditions-from",
        type=Path,
        help="Hard-link matching Stage-2 conditions instead of duplicating prompt embeddings",
    )
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
    reuse_prompts = _reuse_manifest_prompts(args.reuse_conditions_from) if args.reuse_conditions_from else None

    for progress, request in enumerate(requests, start=1):
        expected = _expected_paths(args.output, request.index)
        if args.reuse_conditions_from is not None:
            if reuse_prompts is None or reuse_prompts.get(request.index) != request.prompt:
                raise ValueError(f"reused condition manifest prompt mismatch for index {request.index}")
            _reuse_condition(args.reuse_conditions_from, args.output, request.index)
        generated = expected[:-1] if args.reuse_conditions_from is not None else expected
        present = [path for path in generated if path.exists()]
        if len(present) == len(generated):
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
            save_stage1_trajectory_step(
                args.output,
                _request_index,
                step_index,
                save_conditions=args.reuse_conditions_from is None,
                **values,
            )
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
