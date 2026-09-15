#!/usr/bin/env python3
"""Evaluate one Stage-1 transition checkpoint against its clean base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.stage1_distillation_evaluator import evaluate_stage1_student, load_stage1_student

try:
    from scripts.evaluate_stage1_compressed_curriculum import percent_change
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from evaluate_stage1_compressed_curriculum import percent_change  # type: ignore[no-redef]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    student_model, metadata = load_stage1_student(
        args.model,
        args.checkpoint,
        transformer_file=args.transformer_file,
    )
    student = evaluate_stage1_student(student_model, args.data, metadata, limit=args.limit)
    del student_model
    aggressive_cleanup()

    baseline_model = load_transformer(args.model, transformer_file=args.transformer_file)
    baseline = evaluate_stage1_student(baseline_model, args.data, metadata, limit=args.limit)
    del baseline_model
    aggressive_cleanup()

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "span": [int(metadata["stage1_noise_step_index"]), int(metadata["stage1_noise_step_end_index"])],
        "student": student,
        "baseline": baseline,
        "video_mse_change_percent": percent_change(student["video_mse"], baseline["video_mse"]),
        "audio_mse_change_percent": percent_change(student["audio_mse"], baseline["audio_mse"]),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
