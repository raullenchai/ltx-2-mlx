#!/usr/bin/env python3
"""Evaluate three independent Stage-1 adapters only on their bound spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.stage1_distillation_evaluator import evaluate_stage1_student, load_stage1_student

try:
    from scripts.evaluate_stage1_compressed_curriculum import percent_change
    from scripts.train_stage1_compressed_curriculum import PRIMARY_PHASES, require_phase_checkpoint
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from evaluate_stage1_compressed_curriculum import percent_change  # type: ignore[no-redef]
    from train_stage1_compressed_curriculum import (  # type: ignore[no-redef]
        PRIMARY_PHASES,
        require_phase_checkpoint,
    )


SEGMENT_PHASES = PRIMARY_PHASES[:3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if len(args.checkpoint) != len(SEGMENT_PHASES):
        raise ValueError("exactly three --checkpoint values are required in 0-3, 3-5, 5-7 order")

    students = []
    metadata_by_phase = []
    for checkpoint, phase in zip(args.checkpoint, SEGMENT_PHASES, strict=True):
        path = Path(checkpoint)
        require_phase_checkpoint(path, phase, "span-v2", "independent")
        model, metadata = load_stage1_student(args.model, path, transformer_file=args.transformer_file)
        students.append(evaluate_stage1_student(model, args.data, metadata, limit=args.limit))
        metadata_by_phase.append(metadata)
        del model
        aggressive_cleanup()

    baseline_model = load_transformer(args.model, transformer_file=args.transformer_file)
    baselines = [
        evaluate_stage1_student(baseline_model, args.data, metadata, limit=args.limit)
        for metadata in metadata_by_phase
    ]
    del baseline_model
    aggressive_cleanup()

    transitions = []
    for phase, student, baseline in zip(SEGMENT_PHASES, students, baselines, strict=True):
        transitions.append(
            {
                "name": phase.name,
                "indices": [phase.start_index, phase.target_index],
                "student": student,
                "baseline": baseline,
                "video_mse_change_percent": percent_change(student["video_mse"], baseline["video_mse"]),
                "audio_mse_change_percent": percent_change(student["audio_mse"], baseline["audio_mse"]),
            }
        )
    result = {
        "adapter_mode": "independent",
        "noise_coupling": "span-v2",
        "schedule_indices": [0, 3, 5, 7, 8],
        "transitions": transitions,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
