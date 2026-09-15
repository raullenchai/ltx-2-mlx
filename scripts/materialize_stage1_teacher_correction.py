#!/usr/bin/env python3
"""Roll the clean teacher from a student boundary to a reachable target."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

try:
    from scripts.materialize_stage1_student_rollout import (
        _hardlink_dataset_view,
        _precomputed_root,
        _prompt,
        _sample_index,
        _scalar,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from materialize_stage1_student_rollout import (  # type: ignore[no-redef]
        _hardlink_dataset_view,
        _precomputed_root,
        _prompt,
        _sample_index,
        _scalar,
    )

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.distilled import ANCESTRAL_ETA, ANCESTRAL_S_NOISE
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import ancestral_denoise_loop
from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.stage1_distillation_evaluator import _add_batch
from ltx_trainer_mlx.training_strategies.stage1_transition_distill import (
    Stage1TransitionDistillConfig,
    Stage1TransitionDistillStrategy,
)
from ltx_trainer_mlx.trajectory import save_stage1_trajectory_step


def _strategy(start_step: int, target_step: int) -> Stage1TransitionDistillStrategy:
    total_steps = len(DISTILLED_SIGMAS) - 1
    if not 0 <= start_step < target_step <= total_steps:
        raise ValueError(f"teacher correction span must satisfy 0 <= start < end <= {total_steps}")
    return Stage1TransitionDistillStrategy(
        Stage1TransitionDistillConfig(
            sigma=DISTILLED_SIGMAS[start_step],
            target_sigma=DISTILLED_SIGMAS[target_step],
            video_start_latents_dir=f"stage1_video_step_{start_step:02d}",
            video_terminal_latents_dir=f"stage1_video_step_{target_step:02d}",
            audio_start_latents_dir=f"stage1_audio_step_{start_step:02d}",
            audio_terminal_latents_dir=f"stage1_audio_step_{target_step:02d}",
            ancestral_noise_step_index=start_step,
            ancestral_noise_step_end_index=target_step,
            ancestral_noise_reference_sigmas=DISTILLED_SIGMAS,
            ancestral_noise_total_steps=total_steps,
            ancestral_eta=ANCESTRAL_ETA,
            ancestral_s_noise=ANCESTRAL_S_NOISE,
        )
    )


def _state(latent: mx.array, positions: mx.array, attention_mask: mx.array | None) -> LatentState:
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros_like(latent),
        denoise_mask=mx.ones((*latent.shape[:2], 1), dtype=latent.dtype),
        positions=positions,
        attention_mask=attention_mask,
    )


def teacher_correct_transition(
    model,
    inputs,
    *,
    noise_seed: int,
    start_step: int,
    target_step: int,
) -> tuple[mx.array, mx.array, float]:
    """Run all original teacher steps from the supplied student state."""
    if inputs.audio is None:
        raise ValueError("Stage-1 teacher correction requires audio")
    start = time.perf_counter()
    output = ancestral_denoise_loop(
        model=X0Model(model),
        video_state=_state(inputs.video.latent, inputs.video.positions, inputs.video.attention_mask),
        audio_state=_state(inputs.audio.latent, inputs.audio.positions, inputs.audio.attention_mask),
        video_text_embeds=inputs.video.context,
        audio_text_embeds=inputs.audio.context,
        sigmas=DISTILLED_SIGMAS[start_step : target_step + 1],
        noise_seed=noise_seed,
        noise_step_indices=list(range(start_step, target_step)),
        noise_total_steps=len(DISTILLED_SIGMAS) - 1,
        eta=ANCESTRAL_ETA,
        s_noise=ANCESTRAL_S_NOISE,
        show_progress=False,
    )
    mx.eval(output.video_latent, output.audio_latent)
    return output.video_latent, output.audio_latent, time.perf_counter() - start


def materialize_teacher_correction(
    model_dir: Path,
    source: Path,
    output: Path,
    *,
    start_step: int,
    target_step: int,
    transformer_file: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replace a teacher target with the teacher rollout from student input."""
    strategy = _strategy(start_step, target_step)
    target_sources = {
        strategy.config.video_terminal_latents_dir,
        strategy.config.audio_terminal_latents_dir,
    }
    linked_files = _hardlink_dataset_view(source, output, target_sources)
    marker = output / ".teacher-correction-incomplete.json"
    marker.write_text(json.dumps({"span": [start_step, target_step]}) + "\n")

    dataset = PrecomputedDataset(str(source), data_sources=strategy.get_data_sources())
    count = min(len(dataset), limit) if limit is not None else len(dataset)
    if count <= 0:
        raise ValueError("teacher-correction dataset is empty")
    source_precomputed = _precomputed_root(source).resolve()
    model = load_transformer(model_dir, transformer_file=transformer_file)
    elapsed_total = 0.0
    try:
        for position in range(count):
            relative = dataset.sample_files["video_start"][position]
            index = _sample_index(relative)
            batch = _add_batch(dataset[position])
            inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
            video_data = batch["video_start"]
            video, audio, elapsed = teacher_correct_transition(
                model,
                inputs,
                noise_seed=_scalar(video_data, "noise_seed", int),
                start_step=start_step,
                target_step=target_step,
            )
            elapsed_total += elapsed
            conditions = batch["conditions"]
            source_video = source_precomputed / strategy.config.video_start_latents_dir / relative
            save_stage1_trajectory_step(
                output,
                index,
                target_step,
                sigma=DISTILLED_SIGMAS[target_step],
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
            print(f"materialized teacher correction {position + 1}/{count} (index {index})")
    finally:
        del model
        aggressive_cleanup()

    parent_report = _precomputed_root(source).resolve().parent / "student-rollout.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "capability": "ltx_stage1_teacher_corrected_rollout_dataset_v1",
        "source": str(source.resolve()),
        "model": str(model_dir.resolve()),
        "transformer_file": transformer_file,
        "span": [start_step, target_step],
        "fine_step_count": target_step - start_step,
        "sample_count": count,
        "linked_file_count": linked_files,
        "teacher_seconds": elapsed_total,
    }
    if parent_report.is_file():
        report["student_rollout"] = json.loads(parent_report.read_text())
    report_path = output / "teacher-correction.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    marker.unlink()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = materialize_teacher_correction(
        args.model,
        args.data,
        args.output,
        start_step=args.start_step,
        target_step=args.target_step,
        transformer_file=args.transformer_file,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
