#!/usr/bin/env python3
"""Run the fixed multi-resolution curriculum for a stage-2 3 -> 1 student."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

STAGES = (
    ("468", "buckets/468/train", 100, 1.0e-4, 20),
    ("1536", "buckets/1536/train", 50, 3.0e-5, 10),
    ("3072", "buckets/3072/train", 16, 1.0e-5, 4),
    ("replay", "splits/train", 24, 1.0e-6, 6),
)


def build_config(
    *,
    model: Path,
    transformer_file: str,
    data: Path,
    output: Path,
    steps: int,
    learning_rate: float,
    checkpoint_interval: int,
    load_checkpoint: Path | None,
) -> dict:
    model_config: dict[str, object] = {
        "model_path": str(model),
        "transformer_file": transformer_file,
        "training_mode": "lora",
    }
    if load_checkpoint is not None:
        model_config["load_checkpoint"] = str(load_checkpoint)
    return {
        "model": model_config,
        "lora": {
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["to_q", "to_k", "to_v"],
        },
        "optimization": {
            "learning_rate": learning_rate,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "enable_gradient_checkpointing": True,
            "scheduler_type": "constant" if learning_rate == 1.0e-6 else "linear",
            "scheduler_params": {} if learning_rate == 1.0e-6 else {"start_factor": 1.0, "end_factor": 0.2},
        },
        "data": {"preprocessed_data_root": str(data)},
        "training_strategy": {
            "name": "stage2_terminal_distill",
            "sigma": 0.909375,
            "target_sigma": 0.0,
            "video_terminal_latents_dir": "stage2_video_terminal_latents",
            "audio_terminal_latents_dir": "stage2_audio_terminal_latents",
            "video_loss_weight": 1.0,
            "audio_loss_weight": 1.0,
        },
        "flow_matching": {"timestep_sampling_mode": "uniform"},
        "validation": {"prompts": [], "interval": None, "generate_audio": True},
        "checkpoints": {"interval": checkpoint_interval, "keep_last_n": 6, "precision": "bfloat16"},
        "seed": 42,
        "output_dir": str(output),
    }


def _write_status(root: Path, value: str) -> None:
    temporary = root / ".status.tmp"
    temporary.write_text(value + "\n")
    temporary.replace(root / "status")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint: Path | None = None
    try:
        for name, data_relative, steps, learning_rate, interval in STAGES:
            output = args.output_root / name
            expected = output / "checkpoints" / f"lora_weights_step_{steps:05d}.safetensors"
            if expected.is_file():
                checkpoint = expected
                continue
            data = args.data_root / data_relative
            if not data.is_dir():
                raise FileNotFoundError(f"missing curriculum dataset: {data}")
            output.mkdir(parents=True, exist_ok=True)
            config = build_config(
                model=args.model,
                transformer_file=args.transformer_file,
                data=data,
                output=output,
                steps=steps,
                learning_rate=learning_rate,
                checkpoint_interval=interval,
                load_checkpoint=checkpoint,
            )
            config_path = output / "training-config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            _write_status(args.output_root, f"training-{name}")
            with (output / "train.log").open("a") as log:
                subprocess.run(
                    [sys.executable, "-m", "ltx_pipelines_mlx.cli", "train", "--config", str(config_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not expected.is_file():
                raise FileNotFoundError(f"training completed without expected checkpoint: {expected}")
            checkpoint = expected
    except Exception:
        _write_status(args.output_root, "training-failed")
        raise
    _write_status(args.output_root, "training-complete")
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
