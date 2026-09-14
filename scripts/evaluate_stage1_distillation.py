#!/usr/bin/env python3
"""Evaluate one compressed stage-1 checkpoint on held-out teacher boundaries."""

from __future__ import annotations

import argparse
import json

from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.stage1_distillation_evaluator import evaluate_stage1_student, load_stage1_student


def percent_change(value: float, baseline: float) -> float | None:
    """Return a JSON-safe relative change, or null for a zero baseline."""
    if baseline == 0.0:
        return None
    return (value / baseline - 1.0) * 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    model, metadata = load_stage1_student(
        args.model,
        args.checkpoint,
        transformer_file=args.transformer_file,
    )
    student = evaluate_stage1_student(model, args.data, metadata, limit=args.limit)
    del model
    aggressive_cleanup()
    baseline_model = load_transformer(args.model, transformer_file=args.transformer_file)
    baseline = evaluate_stage1_student(baseline_model, args.data, metadata, limit=args.limit)
    result = {
        "student": student,
        "baseline": baseline,
        "video_mse_change_percent": percent_change(student["video_mse"], baseline["video_mse"]),
        "audio_mse_change_percent": percent_change(student["audio_mse"], baseline["audio_mse"]),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as output:
            output.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
