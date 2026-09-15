#!/usr/bin/env python3
"""Train each learned span-v2 Stage-1 segment independently from the clean base."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

try:
    from scripts.train_stage1_compressed_curriculum import (
        PRIMARY_PHASES,
        _write_status,
        build_config,
        require_phase_checkpoint,
    )
except ModuleNotFoundError:  # Direct script execution.
    from train_stage1_compressed_curriculum import (  # type: ignore[no-redef]
        PRIMARY_PHASES,
        _write_status,
        build_config,
        require_phase_checkpoint,
    )


SEGMENT_PHASES = PRIMARY_PHASES[:3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        for phase in SEGMENT_PHASES:
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
    for phase in SEGMENT_PHASES:
        print(args.output_root / phase.name / "checkpoints" / f"lora_weights_step_{phase.steps:05d}.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
