#!/usr/bin/env python3
"""Render a diagnostic segmented Stage-1 package with selected adapters disabled."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from safetensors import safe_open

from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS

_SPANS = {
    "0-3": (0, 3),
    "3-5": (3, 5),
    "5-7": (5, 7),
}


def apply_diagnostic_base_spans(pipe: DistilledPipeline, spans: tuple[tuple[int, int], ...]) -> None:
    """Configure a validated diagnostic package for adapter attribution."""
    package = pipe._fast_stage1_segmented_package
    if package is None:
        raise ValueError("diagnostic base spans require a segmented fast stage-1 package")
    selected = frozenset(spans)
    available = {(segment.start_index, segment.end_index) for segment in package.segments}
    if not selected.issubset(available):
        raise ValueError("diagnostic base spans must name packaged Stage-1 segments")
    if not package.qualification_revision.startswith("diagnostic-"):
        raise ValueError("diagnostic base spans require a diagnostic qualification")
    pipe._diagnostic_base_stage1_spans = selected


def parse_base_schedule(value: str) -> tuple[int, ...]:
    """Parse a complete fine-step boundary list for a clean-base diagnostic."""
    try:
        boundaries = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("base schedule must be comma-separated integers") from exc
    if len(boundaries) < 3:
        raise argparse.ArgumentTypeError("base schedule requires at least two transitions")
    if boundaries[0] != 0 or boundaries[-1] != len(DISTILLED_SIGMAS) - 1:
        raise argparse.ArgumentTypeError("base schedule must cover fine-step boundaries 0 through 8")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise argparse.ArgumentTypeError("base schedule boundaries must increase strictly")
    return boundaries


def apply_diagnostic_base_schedule(pipe: DistilledPipeline, boundaries: tuple[int, ...]) -> None:
    """Replace packaged spans with an all-clean-base research schedule.

    The terminal transition remains deterministic, matching the production
    segmented path. Adapter paths are retained only as inert schema fields;
    every configured non-terminal span is explicitly selected for clean base.
    """
    package = pipe._fast_stage1_segmented_package
    if package is None:
        raise ValueError("diagnostic base schedule requires a segmented fast stage-1 package")
    if not package.qualification_revision.startswith("diagnostic-"):
        raise ValueError("diagnostic base schedule requires a diagnostic qualification")
    if boundaries[0] != 0 or boundaries[-1] != package.noise_total_steps:
        raise ValueError("diagnostic base schedule must cover the complete reference schedule")
    if len(boundaries) < 3 or any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("diagnostic base schedule boundaries must increase strictly")

    template = package.segments[0]
    learned_spans = tuple(zip(boundaries[:-2], boundaries[1:-1], strict=True))
    segments = tuple(
        replace(
            template,
            start_index=start,
            end_index=end,
            sigma=package.noise_reference_sigmas[start],
            target_sigma=package.noise_reference_sigmas[end],
        )
        for start, end in learned_spans
    )
    pipe._fast_stage1_segmented_package = replace(
        package,
        segments=segments,
        schedule=tuple(package.noise_reference_sigmas[index] for index in boundaries),
    )
    pipe._diagnostic_base_stage1_spans = frozenset(learned_spans)


def apply_diagnostic_adapter_spans(pipe: DistilledPipeline, values: tuple[str, ...]) -> None:
    """Bind explicitly named research checkpoints to a custom base schedule."""
    package = pipe._fast_stage1_segmented_package
    if package is None or not package.qualification_revision.startswith("diagnostic-"):
        raise ValueError("diagnostic adapter spans require a diagnostic package")
    replacements = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or label.count("-") != 1:
            raise ValueError("diagnostic adapter span must use START-END=CHECKPOINT")
        try:
            span = tuple(int(item) for item in label.split("-"))
        except ValueError as exc:
            raise ValueError("diagnostic adapter span indices must be integers") from exc
        if span in replacements:
            raise ValueError("diagnostic adapter spans must be unique")
        path = Path(raw_path)
        if not path.is_file() or path.suffix != ".safetensors":
            raise ValueError("diagnostic adapter checkpoint must be an existing .safetensors file")
        replacements[span] = path

    configured = {(segment.start_index, segment.end_index): segment for segment in package.segments}
    if not replacements.keys() <= configured.keys():
        raise ValueError("diagnostic adapter spans must name configured non-terminal spans")
    for span, path in replacements.items():
        segment = configured[span]
        with safe_open(str(path), framework="numpy") as checkpoint:
            metadata = checkpoint.metadata() or {}
        expected = {
            "distillation": "stage1_transition",
            "stage1_curriculum_noise_coupling": "span-v2",
            "stage1_curriculum_adapter_mode": "independent",
            "stage1_noise_step_index": str(span[0]),
            "stage1_noise_step_end_index": str(span[1]),
            "stage1_sigma": str(segment.sigma),
            "stage1_target_sigma": str(segment.target_sigma),
        }
        if any(metadata.get(key) != wanted for key, wanted in expected.items()):
            raise ValueError(f"diagnostic adapter checkpoint metadata does not match span {span}")
        rank = int(metadata.get("lora_rank", "0"))
        alpha = float(metadata.get("lora_alpha", "0"))
        if rank <= 0 or alpha <= 0:
            raise ValueError("diagnostic adapter checkpoint requires positive LoRA rank and alpha")
        configured[span] = replace(segment, adapter_path=path, lora_rank=rank, lora_alpha=alpha)

    pipe._fast_stage1_segmented_package = replace(
        package,
        segments=tuple(configured[(segment.start_index, segment.end_index)] for segment in package.segments),
    )
    pipe._diagnostic_base_stage1_spans = frozenset(
        span for span in pipe._diagnostic_base_stage1_spans if span not in replacements
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fast-stage1-segmented-manifest", required=True)
    parser.add_argument("--fast-stage2-manifest")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--base-span", action="append", choices=tuple(_SPANS))
    selection.add_argument("--base-schedule", type=parse_base_schedule, metavar="0,1,...,8")
    parser.add_argument("--adapter-span", action="append", default=[], metavar="START-END=CHECKPOINT")
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
    )
    if args.base_schedule is not None:
        apply_diagnostic_base_schedule(pipe, args.base_schedule)
    else:
        if args.adapter_span:
            parser.error("--adapter-span requires --base-schedule")
        apply_diagnostic_base_spans(pipe, tuple(_SPANS[value] for value in args.base_span))
    apply_diagnostic_adapter_spans(pipe, tuple(args.adapter_span))
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
