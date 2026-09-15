#!/usr/bin/env python3
"""Render a diagnostic segmented Stage-1 package with selected adapters disabled."""

from __future__ import annotations

import argparse
from pathlib import Path

from ltx_pipelines_mlx.distilled import DistilledPipeline

_SPANS = {
    "0-3": (0, 3),
    "3-5": (3, 5),
    "5-7": (5, 7),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fast-stage1-segmented-manifest", required=True)
    parser.add_argument("--fast-stage2-manifest")
    parser.add_argument("--base-span", action="append", choices=tuple(_SPANS), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=241)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-ram", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    pipe = DistilledPipeline(
        model_dir=args.model,
        low_memory=True,
        low_ram_streaming=args.low_ram,
        fast_stage1_segmented_manifest=args.fast_stage1_segmented_manifest,
        fast_stage2_manifest=args.fast_stage2_manifest,
        diagnostic_base_stage1_spans=tuple(_SPANS[value] for value in args.base_span),
    )
    pipe.verbose = not args.quiet
    pipe.generate_and_save(
        prompt=args.prompt,
        output_path=args.output,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
