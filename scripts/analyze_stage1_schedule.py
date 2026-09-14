#!/usr/bin/env python3
"""Rank four-evaluation stage-1 schedules from captured ancestral paths."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mlx.core as mx


def chord_error(path: list[mx.array], sigmas: list[float], boundaries: tuple[int, ...]) -> float:
    """Mean relative RMSE of skipped boundaries against sigma-linear chords."""
    errors: list[float] = []
    for start, end in zip(boundaries, boundaries[1:]):
        for middle in range(start + 1, end):
            fraction = (sigmas[middle] - sigmas[end]) / (sigmas[start] - sigmas[end])
            predicted = path[end].astype(mx.float32) + fraction * (
                path[start].astype(mx.float32) - path[end].astype(mx.float32)
            )
            target = path[middle].astype(mx.float32)
            rmse = mx.sqrt(mx.mean(mx.square(predicted - target)))
            target_rms = mx.sqrt(mx.mean(mx.square(target)))
            errors.append(float((rmse / mx.maximum(target_rms, 1e-12)).item()))
    return sum(errors) / len(errors) if errors else 0.0


def _load_path(root: Path, prefix: str, index: int) -> tuple[list[mx.array], list[float]]:
    path, sigmas = [], []
    for step in range(9):
        values = mx.load(
            str(root / ".precomputed" / f"stage1_{prefix}_step_{step:02d}" / f"latent_{index:04d}.safetensors")
        )
        path.append(values["latents"])
        sigmas.append(float(values["sigma"].reshape(-1)[0].item()))
    return path, sigmas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    starts = sorted((args.data / ".precomputed" / "stage1_video_step_00").glob("latent_*.safetensors"))
    if args.limit is not None:
        starts = starts[: args.limit]
    if not starts:
        raise ValueError("stage-1 trajectory dataset is empty")

    candidates = [(0, *choice, 7, 8) for choice in itertools.combinations(range(1, 7), 2)]
    aggregate = {candidate: {"video": [], "audio": []} for candidate in candidates}
    sigmas: list[float] | None = None
    for start in starts:
        index = int(start.stem.removeprefix("latent_"))
        video_path, current_sigmas = _load_path(args.data, "video", index)
        audio_path, audio_sigmas = _load_path(args.data, "audio", index)
        if audio_sigmas != current_sigmas or (sigmas is not None and current_sigmas != sigmas):
            raise ValueError("captured trajectories do not share one sigma schedule")
        sigmas = current_sigmas
        for candidate in candidates:
            aggregate[candidate]["video"].append(chord_error(video_path, sigmas, candidate))
            aggregate[candidate]["audio"].append(chord_error(audio_path, sigmas, candidate))

    assert sigmas is not None
    rankings = []
    for candidate, metrics in aggregate.items():
        video = sum(metrics["video"]) / len(metrics["video"])
        audio = sum(metrics["audio"]) / len(metrics["audio"])
        rankings.append(
            {
                "boundary_indices": list(candidate),
                "sigmas": [sigmas[index] for index in candidate],
                "video_chord_relative_rmse": video,
                "audio_chord_relative_rmse": audio,
                "mean_chord_relative_rmse": (video + audio) / 2,
            }
        )
    rankings.sort(key=lambda item: item["mean_chord_relative_rmse"])
    result = {"samples": len(starts), "diagnostic_only": True, "rankings": rankings}
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
