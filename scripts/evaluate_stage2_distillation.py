#!/usr/bin/env python3
"""Evaluate a terminal-distillation LoRA on paired teacher trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_trainer_mlx.distillation_evaluator import (
    evaluate_terminal_student,
    load_terminal_baseline,
    load_terminal_student,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", help="Terminal LoRA; omit to measure the unadapted one-step baseline.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--sigma", type=float, default=0.909375, help="Baseline start sigma (ignored with checkpoint).")
    parser.add_argument("--target-sigma", type=float, default=0.0, help="Baseline target sigma.")
    parser.add_argument("--video-target-dir", default="stage2_video_terminal_latents")
    parser.add_argument("--audio-target-dir", default="stage2_audio_terminal_latents")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.checkpoint:
        model, metadata = load_terminal_student(
            args.model,
            args.checkpoint,
            transformer_file=args.transformer_file,
        )
    else:
        model, metadata = load_terminal_baseline(
            args.model,
            sigma=args.sigma,
            target_sigma=args.target_sigma,
            video_target_latents_dir=args.video_target_dir,
            audio_target_latents_dir=args.audio_target_dir,
            transformer_file=args.transformer_file,
        )
    result = evaluate_terminal_student(model, args.data, metadata, limit=args.limit)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
