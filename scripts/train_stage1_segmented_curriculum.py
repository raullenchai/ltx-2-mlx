#!/usr/bin/env python3
"""Train each learned span-v2 Stage-1 segment independently from the clean base."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import yaml

try:
    from scripts.train_stage1_compressed_curriculum import (
        PRIMARY_PHASES,
        Phase,
        _write_status,
        build_config,
        require_phase_checkpoint,
    )
except ModuleNotFoundError:  # Direct script execution.
    from train_stage1_compressed_curriculum import (  # type: ignore[no-redef]
        PRIMARY_PHASES,
        Phase,
        _write_status,
        build_config,
        require_phase_checkpoint,
    )


SEGMENT_PHASES = PRIMARY_PHASES[:3]
DIAGNOSTIC_PHASES = (Phase("diagnostic-1-3", 1, 3, 0.99375, 0.98125, 1, 100, 5.0e-5, 20),)
AVAILABLE_PHASES = SEGMENT_PHASES + DIAGNOSTIC_PHASES


def select_segment_phases(spans: list[str] | None, steps: int | None = None) -> tuple[Phase, ...]:
    """Select independent spans and optionally override one control's budget."""
    selected = set(spans or ())
    phases = tuple(
        phase
        for phase in (AVAILABLE_PHASES if selected else SEGMENT_PHASES)
        if not selected or f"{phase.start_index}-{phase.target_index}" in selected
    )
    if steps is None:
        return phases
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if len(selected) != 1 or len(phases) != 1:
        raise ValueError("--steps requires exactly one --span")
    return (replace(phases[0], steps=steps, checkpoint_interval=steps),)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    parser.add_argument(
        "--span",
        action="append",
        choices=tuple(f"{phase.start_index}-{phase.target_index}" for phase in AVAILABLE_PHASES),
        help="Train only selected span(s), including diagnostic spans (default: three product spans)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the training budget for exactly one selected span",
    )
    args = parser.parse_args()

    try:
        phases = select_segment_phases(args.span, args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        for phase in phases:
            output = args.output_root / phase.name
            expected = output / "checkpoints" / f"lora_weights_step_{phase.steps:05d}.safetensors"
            if expected.is_file():
                require_phase_checkpoint(expected, phase, "span-v2", "independent")
                continue
            output.mkdir(parents=True, exist_ok=True)
            config = build_config(
                phase=phase,
                model=args.model,
                transformer_file=args.transformer_file,
                data=args.data,
                output=output,
                load_checkpoint=None,
                noise_coupling="span-v2",
                adapter_mode="independent",
            )
            # Keep only the final checkpoint. Three 160 MB artifacts fit the
            # constrained qualification host without accumulating snapshots.
            config["checkpoints"] = {
                "interval": phase.steps,
                "keep_last_n": 1,
                "precision": "bfloat16",
            }
            config_path = output / "training-config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            _write_status(args.output_root, f"training-{phase.name}")
            with (output / "train.log").open("a") as log:
                subprocess.run(
                    [sys.executable, "-m", "ltx_pipelines_mlx.cli", "train", "--config", str(config_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            require_phase_checkpoint(expected, phase, "span-v2", "independent")
    except Exception:
        _write_status(args.output_root, "training-failed")
        raise

    _write_status(args.output_root, "training-complete")
    for phase in phases:
        print(args.output_root / phase.name / "checkpoints" / f"lora_weights_step_{phase.steps:05d}.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
