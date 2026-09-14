#!/usr/bin/env python3
"""Train one shared adapter for the selected four-transition stage-1 schedule."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from safetensors import safe_open


@dataclass(frozen=True)
class Phase:
    name: str
    start_index: int
    target_index: int
    sigma: float
    target_sigma: float
    noise_step_index: int | None
    steps: int
    learning_rate: float
    checkpoint_interval: int


# Boundaries [0, 3, 5, 7, 8] were independently selected on the training and
# held-out trajectory sets. The final 7 -> 8 transition is deterministic
# because its target is sigma zero and therefore draws no ancestral noise.
PRIMARY_PHASES = (
    Phase("primary-0-3", 0, 3, 1.0, 0.98125, 0, 100, 5.0e-5, 20),
    Phase("primary-3-5", 3, 5, 0.98125, 0.909375, 3, 80, 3.0e-5, 20),
    Phase("primary-5-7", 5, 7, 0.909375, 0.421875, 5, 80, 1.0e-5, 20),
    Phase("primary-7-8", 7, 8, 0.421875, 0.0, None, 40, 3.0e-6, 10),
)

# Sequential training is deliberately followed by a complete, low-LR replay
# pass. This does not prove that catastrophic forgetting is absent; the paired
# evaluator must still gate every transition independently.
REPLAY_PHASES = tuple(
    Phase(
        f"replay-{phase.start_index}-{phase.target_index}",
        phase.start_index,
        phase.target_index,
        phase.sigma,
        phase.target_sigma,
        phase.noise_step_index,
        24,
        1.0e-6,
        6,
    )
    for phase in PRIMARY_PHASES
)
PHASES = PRIMARY_PHASES + REPLAY_PHASES


def validate_phase_checkpoint(path: Path, phase: Phase) -> None:
    """Reject a stale checkpoint whose metadata does not match its phase."""
    with safe_open(path, framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
    expected = {
        "distillation": "stage1_transition",
        "stage1_sigma": str(phase.sigma),
        "stage1_target_sigma": str(phase.target_sigma),
        "stage1_video_start_latents_dir": f"stage1_video_step_{phase.start_index:02d}",
        "stage1_video_target_latents_dir": f"stage1_video_step_{phase.target_index:02d}",
        "stage1_audio_start_latents_dir": f"stage1_audio_step_{phase.start_index:02d}",
        "stage1_audio_target_latents_dir": f"stage1_audio_step_{phase.target_index:02d}",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint {path} does not match {phase.name}: {key}")
    if int(metadata.get("lora_rank", "0")) <= 0 or float(metadata.get("lora_alpha", "0")) <= 0:
        raise ValueError(f"checkpoint {path} does not declare a valid LoRA scale")
    if phase.noise_step_index is None:
        if "stage1_noise_step_index" in metadata or metadata.get("stage1_sampler") == "ancestral":
            raise ValueError(f"checkpoint {path} incorrectly declares terminal ancestral noise")
    elif (
        metadata.get("stage1_sampler") != "ancestral"
        or int(metadata.get("stage1_noise_step_index", "-1")) != phase.noise_step_index
        or int(metadata.get("stage1_noise_total_steps", "0")) != 8
    ):
        raise ValueError(f"checkpoint {path} does not match {phase.name}: ancestral noise lane")


def build_config(
    *,
    phase: Phase,
    model: Path,
    transformer_file: str,
    data: Path,
    output: Path,
    load_checkpoint: Path | None,
) -> dict:
    model_config: dict[str, object] = {
        "model_path": str(model),
        "transformer_file": transformer_file,
        "training_mode": "lora",
    }
    if load_checkpoint is not None:
        model_config["load_checkpoint"] = str(load_checkpoint)

    strategy: dict[str, object] = {
        "name": "stage1_transition_distill",
        "sigma": phase.sigma,
        "target_sigma": phase.target_sigma,
        "video_start_latents_dir": f"stage1_video_step_{phase.start_index:02d}",
        "video_terminal_latents_dir": f"stage1_video_step_{phase.target_index:02d}",
        "audio_start_latents_dir": f"stage1_audio_step_{phase.start_index:02d}",
        "audio_terminal_latents_dir": f"stage1_audio_step_{phase.target_index:02d}",
        "conditions_dir": "stage1_conditions",
        "video_loss_weight": 1.0,
        "audio_loss_weight": 1.0,
    }
    if phase.noise_step_index is not None:
        strategy.update(
            ancestral_noise_step_index=phase.noise_step_index,
            ancestral_noise_total_steps=8,
            ancestral_eta=1.0,
            ancestral_s_noise=1.0,
        )

    return {
        "model": model_config,
        "lora": {
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": ["to_q", "to_k", "to_v"],
        },
        "optimization": {
            "learning_rate": phase.learning_rate,
            "steps": phase.steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "enable_gradient_checkpointing": True,
            "scheduler_type": "constant" if phase.learning_rate == 1.0e-6 else "linear",
            "scheduler_params": {} if phase.learning_rate == 1.0e-6 else {"start_factor": 1.0, "end_factor": 0.2},
        },
        "data": {"preprocessed_data_root": str(data)},
        "training_strategy": strategy,
        "flow_matching": {"timestep_sampling_mode": "uniform"},
        "validation": {"prompts": [], "interval": None, "generate_audio": True},
        "checkpoints": {"interval": phase.checkpoint_interval, "keep_last_n": 5, "precision": "bfloat16"},
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint: Path | None = None
    try:
        for phase in PHASES:
            output = args.output_root / phase.name
            expected = output / "checkpoints" / f"lora_weights_step_{phase.steps:05d}.safetensors"
            if expected.is_file():
                validate_phase_checkpoint(expected, phase)
                checkpoint = expected
                continue
            output.mkdir(parents=True, exist_ok=True)
            config = build_config(
                phase=phase,
                model=args.model,
                transformer_file=args.transformer_file,
                data=args.data,
                output=output,
                load_checkpoint=checkpoint,
            )
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
            if not expected.is_file():
                raise FileNotFoundError(f"training completed without expected checkpoint: {expected}")
            validate_phase_checkpoint(expected, phase)
            checkpoint = expected
    except Exception:
        _write_status(args.output_root, "training-failed")
        raise

    _write_status(args.output_root, "training-complete")
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
