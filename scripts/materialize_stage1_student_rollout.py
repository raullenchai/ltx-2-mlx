#!/usr/bin/env python3
"""Materialize one Stage-1 student span as an on-policy dataset boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors import safe_open

from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.stage1_distillation_evaluator import (
    _add_batch,
    _strategy,
    load_stage1_student,
    predict_stage1_transition,
    read_stage1_checkpoint_metadata,
)
from ltx_trainer_mlx.trajectory import save_stage1_trajectory_step

_STEP_SOURCE = re.compile(r"^stage1_(video|audio)_step_(\d+)$")
_LATENT_FILE = re.compile(r"^latent_(\d+)\.safetensors$")


def _precomputed_root(root: Path) -> Path:
    return root if root.name == ".precomputed" else root / ".precomputed"


def _paired_step(metadata: dict[str, str], boundary: str) -> int:
    sources = (
        metadata[f"stage1_video_{boundary}_latents_dir"],
        metadata[f"stage1_audio_{boundary}_latents_dir"],
    )
    parsed = []
    for source in sources:
        match = _STEP_SOURCE.fullmatch(source)
        if match is None:
            raise ValueError(f"checkpoint target is not a captured Stage-1 boundary: {source}")
        parsed.append((match.group(1), int(match.group(2))))
    if parsed[0] != ("video", parsed[0][1]) or parsed[1] != ("audio", parsed[1][1]):
        raise ValueError("checkpoint video/audio target source modalities are invalid")
    if parsed[0][1] != parsed[1][1]:
        raise ValueError(f"checkpoint video/audio {boundary} steps do not match")
    return parsed[0][1]


def _target_step(metadata: dict[str, str]) -> int:
    start_step = _paired_step(metadata, "start")
    target_step = _paired_step(metadata, "target")
    if target_step <= start_step:
        raise ValueError("checkpoint target step must follow its start step")
    start_index = metadata.get("stage1_noise_step_index")
    if start_index is not None and int(start_index) != start_step:
        raise ValueError("checkpoint start source and span start index do not match")
    end_index = metadata.get("stage1_noise_step_end_index")
    if end_index is not None and int(end_index) != target_step:
        raise ValueError("checkpoint target source and span end index do not match")
    return target_step


def _hardlink_dataset_view(source: Path, output: Path, excluded_sources: set[str]) -> int:
    """Clone a dataset using hard links while leaving selected sources empty."""
    source_precomputed = _precomputed_root(source).resolve()
    if not source_precomputed.is_dir():
        raise FileNotFoundError(f"source precomputed directory not found: {source_precomputed}")
    source_dataset = source_precomputed.parent
    resolved_output = output.resolve()
    if resolved_output == source_dataset or source_dataset in resolved_output.parents:
        raise ValueError("rollout output must not be inside the source dataset")
    if output.exists():
        raise FileExistsError(f"refusing to merge into existing rollout dataset: {output}")
    output_precomputed = output / ".precomputed"
    output_precomputed.mkdir(parents=True)
    if source_precomputed.stat().st_dev != output_precomputed.stat().st_dev:
        raise OSError("source and rollout output must share a filesystem for hard links")

    linked = 0
    for path in sorted(source_precomputed.rglob("*")):
        relative = path.relative_to(source_precomputed)
        if relative.parts and relative.parts[0] in excluded_sources:
            continue
        destination = output_precomputed / relative
        if path.is_symlink():
            raise ValueError(f"dataset symlinks are not accepted: {path}")
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)
            if path.stat().st_ino != destination.stat().st_ino:
                raise RuntimeError(f"hard-link identity check failed: {destination}")
            linked += 1
        else:
            raise ValueError(f"unsupported dataset entry: {path}")

    source_root = source_precomputed.parent
    if source.name != ".precomputed":
        for path in sorted(source_root.iterdir()):
            if path.name == ".precomputed":
                continue
            if path.name in {
                ".rollout-incomplete.json",
                ".teacher-correction-incomplete.json",
                "student-rollout.json",
                "teacher-correction.json",
            }:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            destination = output / path.name
            os.link(path, destination)
            linked += 1
    return linked


def _scalar(data: dict[str, Any], key: str, cast):
    value = data.get(key)
    if not isinstance(value, mx.array) or value.size != 1:
        raise ValueError(f"rollout source requires scalar {key} metadata")
    return cast(value.item())


def _sample_index(relative_path: Path) -> int:
    if len(relative_path.parts) != 1:
        raise ValueError(f"nested Stage-1 sample paths are unsupported: {relative_path}")
    match = _LATENT_FILE.fullmatch(relative_path.name)
    if match is None:
        raise ValueError(f"invalid Stage-1 sample filename: {relative_path.name}")
    return int(match.group(1))


def _prompt(source_file: Path) -> str:
    with safe_open(str(source_file), framework="numpy") as tensors:
        prompt = (tensors.metadata() or {}).get("prompt")
    if not prompt:
        raise ValueError(f"source video boundary has no prompt metadata: {source_file}")
    return prompt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_rollout(
    model_dir: Path,
    checkpoint: Path,
    source: Path,
    output: Path,
    *,
    transformer_file: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a student on every source item and replace only its target boundary."""
    metadata = read_stage1_checkpoint_metadata(checkpoint)
    strategy = _strategy(metadata)
    target_step = _target_step(metadata)
    target_sources = {
        metadata["stage1_video_target_latents_dir"],
        metadata["stage1_audio_target_latents_dir"],
    }
    linked_files = _hardlink_dataset_view(source, output, target_sources)
    marker = output / ".rollout-incomplete.json"
    marker.write_text(json.dumps({"checkpoint": str(checkpoint), "target_step": target_step}) + "\n")

    dataset = PrecomputedDataset(str(source), data_sources=strategy.get_data_sources())
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    if count <= 0:
        raise ValueError("rollout dataset is empty")
    source_precomputed = _precomputed_root(source).resolve()
    elapsed_total = 0.0
    model, _ = load_stage1_student(model_dir, checkpoint, transformer_file=transformer_file)
    try:
        for position in range(count):
            relative = dataset.sample_files["video_start"][position]
            index = _sample_index(relative)
            batch = _add_batch(dataset[position])
            inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
            video, audio, elapsed = predict_stage1_transition(model, strategy, inputs, batch)
            elapsed_total += elapsed
            video_data = batch["video_start"]
            conditions = batch["conditions"]
            source_video = source_precomputed / metadata["stage1_video_start_latents_dir"] / relative
            save_stage1_trajectory_step(
                output,
                index,
                target_step,
                sigma=float(metadata["stage1_target_sigma"]),
                video=video,
                audio=audio,
                video_text_embeds=conditions["video_prompt_embeds"],
                audio_text_embeds=conditions["audio_prompt_embeds"],
                spatial_dims=(
                    _scalar(video_data, "num_frames", int),
                    _scalar(video_data, "height", int),
                    _scalar(video_data, "width", int),
                ),
                frame_rate=_scalar(video_data, "fps", float),
                noise_seed=_scalar(video_data, "noise_seed", int),
                seed=_scalar(video_data, "seed", int),
                prompt=_prompt(source_video),
                save_conditions=False,
            )
            print(f"materialized rollout {position + 1}/{count} (index {index})")
    finally:
        del model
        aggressive_cleanup()

    report = {
        "schema_version": 1,
        "capability": "ltx_stage1_student_rollout_dataset_v1",
        "source": str(source.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "span": [
            _paired_step(metadata, "start"),
            target_step,
        ],
        "sample_count": count,
        "linked_file_count": linked_files,
        "student_seconds": elapsed_total,
    }
    parent_report = source_precomputed.parent / "student-rollout.json"
    if parent_report.is_file():
        report["parent_rollout"] = json.loads(parent_report.read_text())
    report_path = output / "student-rollout.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    marker.unlink()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = materialize_rollout(
        args.model,
        args.checkpoint,
        args.data,
        args.output,
        transformer_file=args.transformer_file,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
