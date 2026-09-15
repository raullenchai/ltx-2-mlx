#!/usr/bin/env python3
"""Evaluate one shared stage-1 adapter independently on all four transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.stage1_distillation_evaluator import evaluate_stage1_student, load_stage1_student
from scripts.train_stage1_compressed_curriculum import PRIMARY_PHASES, Phase


def percent_change(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return (value / baseline - 1.0) * 100


def metadata_for_phase(checkpoint_metadata: dict[str, str], phase: Phase) -> dict[str, str]:
    """Describe one schedule transition while retaining checkpoint LoRA metadata."""
    metadata = dict(checkpoint_metadata)
    span_v2 = metadata.get("stage1_curriculum_noise_coupling") == "span-v2"
    metadata.update(
        distillation="stage1_transition",
        stage1_sigma=str(phase.sigma),
        stage1_target_sigma=str(phase.target_sigma),
        stage1_video_start_latents_dir=f"stage1_video_step_{phase.start_index:02d}",
        stage1_video_target_latents_dir=f"stage1_video_step_{phase.target_index:02d}",
        stage1_audio_start_latents_dir=f"stage1_audio_step_{phase.start_index:02d}",
        stage1_audio_target_latents_dir=f"stage1_audio_step_{phase.target_index:02d}",
        stage1_conditions_dir="stage1_conditions",
    )
    for key in (
        "stage1_sampler",
        "stage1_noise_step_index",
        "stage1_noise_total_steps",
        "stage1_ancestral_eta",
        "stage1_ancestral_s_noise",
        "stage1_noise_step_end_index",
        "stage1_noise_reference_sigmas",
    ):
        metadata.pop(key, None)
    if phase.noise_step_index is not None:
        metadata.update(
            stage1_sampler="ancestral_span_v2" if span_v2 else "ancestral",
            stage1_noise_step_index=str(phase.noise_step_index),
            stage1_noise_total_steps="8",
            stage1_ancestral_eta="1.0",
            stage1_ancestral_s_noise="1.0",
        )
        if span_v2:
            metadata.update(
                stage1_noise_step_end_index=str(phase.target_index),
                stage1_noise_reference_sigmas=json.dumps(DISTILLED_SIGMAS),
            )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    student_model, checkpoint_metadata = load_stage1_student(
        args.model,
        args.checkpoint,
        transformer_file=args.transformer_file,
    )
    student = []
    for phase in PRIMARY_PHASES:
        metadata = metadata_for_phase(checkpoint_metadata, phase)
        student.append(evaluate_stage1_student(student_model, args.data, metadata, limit=args.limit))
    del student_model
    aggressive_cleanup()

    baseline_model = load_transformer(args.model, transformer_file=args.transformer_file)
    baseline = []
    for phase in PRIMARY_PHASES:
        metadata = metadata_for_phase(checkpoint_metadata, phase)
        baseline.append(evaluate_stage1_student(baseline_model, args.data, metadata, limit=args.limit))

    transitions = []
    for phase, student_result, baseline_result in zip(PRIMARY_PHASES, student, baseline, strict=True):
        transitions.append(
            {
                "name": phase.name,
                "indices": [phase.start_index, phase.target_index],
                "student": student_result,
                "baseline": baseline_result,
                "video_mse_change_percent": percent_change(student_result["video_mse"], baseline_result["video_mse"]),
                "audio_mse_change_percent": percent_change(student_result["audio_mse"], baseline_result["audio_mse"]),
            }
        )
    result = {
        "schedule_indices": [0, 3, 5, 7, 8],
        "schedule_sigmas": [1.0, 0.98125, 0.909375, 0.421875, 0.0],
        "transitions": transitions,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
