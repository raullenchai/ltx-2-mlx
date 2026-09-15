#!/usr/bin/env python3
"""Run an experimental alternating DMD2 pilot for the Stage-1 3->7 adapter."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_trainer_mlx.config import LtxTrainerConfig
from ltx_trainer_mlx.gpu_utils import get_gpu_memory_gb
from ltx_trainer_mlx.stage1_distribution_matching import (
    Stage1DmdWeights,
    fake_score_objective,
    generate_stage1_clean,
    generator_objective,
)
from ltx_trainer_mlx.trainer import LtxvTrainer

try:
    from scripts.train_stage1_compressed_curriculum import build_config
    from scripts.train_stage1_segmented_curriculum import DIAGNOSTIC_PHASES
except ModuleNotFoundError:  # Direct script execution.
    from train_stage1_compressed_curriculum import build_config  # type: ignore[no-redef]
    from train_stage1_segmented_curriculum import DIAGNOSTIC_PHASES  # type: ignore[no-redef]


PHASE_3_TO_7 = next(phase for phase in DIAGNOSTIC_PHASES if phase.name == "diagnostic-3-7")
DEFAULT_SCORE_SIGMAS = tuple(DISTILLED_SIGMAS[1:-1])


def build_dmd_config(
    *,
    model: Path,
    transformer_file: str,
    data: Path,
    output: Path,
    checkpoint: Path | None,
    steps: int,
    rank: int,
    learning_rate: float,
    target_modules: list[str] | None,
) -> dict:
    """Build the ordinary trainer shell used by each DMD model."""
    if steps <= 0 or rank <= 0 or learning_rate <= 0:
        raise ValueError("steps, rank, and learning rate must be positive")
    config = build_config(
        phase=PHASE_3_TO_7,
        model=model,
        transformer_file=transformer_file,
        data=data,
        output=output,
        load_checkpoint=checkpoint,
        noise_coupling="span-v2",
        adapter_mode="independent",
        target_modules=target_modules,
    )
    config["lora"].update(rank=rank, alpha=rank)
    config["optimization"].update(
        steps=steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=1,
        scheduler_type="constant",
        scheduler_params={},
    )
    config["checkpoints"] = {"interval": steps, "keep_last_n": 1, "precision": "bfloat16"}
    return config


def sample_score_sigma(rng: random.Random, choices: tuple[float, ...] = DEFAULT_SCORE_SIGMAS) -> float:
    """Sample a non-terminal point from the native LTX distilled schedule."""
    if not choices or any(not 0 < sigma < 1 for sigma in choices):
        raise ValueError("score sigmas must be strictly between zero and one")
    return rng.choice(choices)


def _next_batch(data_iter, dataloader):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def _clip_and_update(model, optimizer, grads, max_grad_norm: float) -> None:
    if max_grad_norm > 0:
        grads, _ = optim.clip_grad_norm(grads, max_norm=max_grad_norm)
    optimizer.update(model, grads)
    mx.eval(optimizer.state, model.parameters())


def train(args: argparse.Namespace) -> tuple[Path, Path]:
    """Execute alternating generator/fake-score turns and save both adapters."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generator_config = build_dmd_config(
        model=args.model,
        transformer_file=args.transformer_file,
        data=args.data,
        output=output / "generator",
        checkpoint=args.checkpoint,
        steps=args.steps,
        rank=args.rank,
        learning_rate=args.generator_lr,
        target_modules=args.target_modules,
    )
    fake_config = build_dmd_config(
        model=args.model,
        transformer_file=args.transformer_file,
        data=args.data,
        output=output / "fake-score",
        checkpoint=args.fake_checkpoint,
        steps=args.steps,
        rank=args.fake_rank,
        learning_rate=args.fake_lr,
        target_modules=args.target_modules,
    )
    (output / "dmd-config.json").write_text(
        json.dumps(
            {
                "generator": generator_config,
                "fake_score": fake_config,
                "paired_weight": args.paired_weight,
                "dm_weight": args.dm_weight,
                "score_sigmas": DEFAULT_SCORE_SIGMAS,
                "fake_warmup_steps": args.fake_warmup_steps,
                "fake_updates_per_generator": args.fake_updates,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    generator_trainer = LtxvTrainer(LtxTrainerConfig(**generator_config))
    fake_trainer = LtxvTrainer(LtxTrainerConfig(**fake_config))
    generator = generator_trainer._transformer
    fake_score = fake_trainer._transformer
    strategy = generator_trainer._training_strategy
    generator_trainer._init_dataloader()
    dataloader = generator_trainer._dataloader
    data_iter = iter(dataloader)

    generator_optimizer = optim.AdamW(args.generator_lr, weight_decay=0.0)
    fake_optimizer = optim.AdamW(args.fake_lr, weight_decay=0.0)
    weights = Stage1DmdWeights(args.paired_weight, args.dm_weight)
    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    shared_metadata = {
        "distillation_objective": "dmd2_rectified_flow_v1",
        "dmd_generator_transition": "3-7",
        "dmd_terminal_transition": "7-8_base",
        "dmd_paired_weight": args.paired_weight,
        "dmd_distribution_matching_weight": args.dm_weight,
        "dmd_score_sigmas": json.dumps(DEFAULT_SCORE_SIGMAS),
        "dmd_fake_warmup_steps": args.fake_warmup_steps,
        "dmd_fake_updates_per_generator": args.fake_updates,
    }
    generator_trainer._checkpoint_metadata_overrides = {**shared_metadata, "dmd_role": "generator"}
    fake_trainer._checkpoint_metadata_overrides = {**shared_metadata, "dmd_role": "fake_score"}

    def generator_loss(inputs, batch, sigma, video_noise, audio_noise):
        return generator_objective(
            generator,
            fake_score,
            strategy,
            inputs,
            batch,
            sigma,
            video_noise,
            audio_noise,
            weights,
        ).total

    generator_value_and_grad = nn.value_and_grad(generator, generator_loss)

    def fake_loss(generated, inputs, sigma, video_noise, audio_noise):
        return fake_score_objective(fake_score, generated, inputs, sigma, video_noise, audio_noise)

    fake_value_and_grad = nn.value_and_grad(fake_score, fake_loss)
    metrics_path = output / "metrics.jsonl"
    start = time.monotonic()
    peak_memory = get_gpu_memory_gb()
    fake_updates_done = 0
    last_generator_checkpoint: Path | None = None
    last_fake_checkpoint: Path | None = None
    last_generator_saved_step = -1
    last_fake_saved_update = -1

    for warmup_step in range(1, args.fake_warmup_steps + 1):
        batch, data_iter = _next_batch(data_iter, dataloader)
        inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
        generated = generate_stage1_clean(generator, strategy, inputs, batch)
        mx.eval(generated.video, generated.audio)
        sigma = sample_score_sigma(rng)
        video_noise = mx.random.normal(generated.video.shape)
        audio_noise = mx.random.normal(generated.audio.shape)
        fake_loss_value, fake_grads = fake_value_and_grad(
            generated,
            inputs,
            sigma,
            video_noise,
            audio_noise,
        )
        mx.eval(fake_loss_value, fake_grads)
        _clip_and_update(fake_score, fake_optimizer, fake_grads, args.max_grad_norm)
        fake_updates_done += 1
        peak_memory = max(peak_memory, get_gpu_memory_gb())
        record = {
            "phase": "fake_warmup",
            "step": warmup_step,
            "paired_anchor_loss": float(generated.paired_loss.item()),
            "fake_score_loss": float(fake_loss_value.item()),
            "score_sigma": sigma,
            "elapsed_seconds": time.monotonic() - start,
            "peak_memory_gb": peak_memory,
        }
        with metrics_path.open("a") as metrics:
            metrics.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.checkpoint_interval and warmup_step % args.checkpoint_interval == 0:
            fake_trainer._global_step = fake_updates_done
            last_fake_checkpoint = fake_trainer._save_checkpoint()
            last_fake_saved_update = fake_updates_done

    for step in range(1, args.steps + 1):
        batch, data_iter = _next_batch(data_iter, dataloader)
        inputs = strategy.prepare_training_inputs(batch, sigma_sampler=None)
        if inputs.audio is None:
            raise ValueError("Stage-1 DMD requires audio")
        sigma = sample_score_sigma(rng)
        video_noise = mx.random.normal(inputs.video.latent.shape)
        audio_noise = mx.random.normal(inputs.audio.latent.shape)
        loss, generator_grads = generator_value_and_grad(inputs, batch, sigma, video_noise, audio_noise)
        mx.eval(loss, generator_grads)
        _clip_and_update(generator, generator_optimizer, generator_grads, args.max_grad_norm)

        fake_loss_value = mx.array(0.0)
        for _ in range(args.fake_updates):
            generated = generate_stage1_clean(generator, strategy, inputs, batch)
            mx.eval(generated.video, generated.audio)
            fake_sigma = sample_score_sigma(rng)
            fake_video_noise = mx.random.normal(generated.video.shape)
            fake_audio_noise = mx.random.normal(generated.audio.shape)
            fake_loss_value, fake_grads = fake_value_and_grad(
                generated,
                inputs,
                fake_sigma,
                fake_video_noise,
                fake_audio_noise,
            )
            mx.eval(fake_loss_value, fake_grads)
            _clip_and_update(fake_score, fake_optimizer, fake_grads, args.max_grad_norm)
            fake_updates_done += 1

        peak_memory = max(peak_memory, get_gpu_memory_gb())
        record = {
            "phase": "alternating",
            "step": step,
            "generator_loss": float(loss.item()),
            "paired_anchor_loss": float(generated.paired_loss.item()),
            "fake_score_loss": float(fake_loss_value.item()),
            "score_sigma": sigma,
            "elapsed_seconds": time.monotonic() - start,
            "peak_memory_gb": peak_memory,
        }
        with metrics_path.open("a") as metrics:
            metrics.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.checkpoint_interval and step % args.checkpoint_interval == 0:
            generator_trainer._global_step = step
            fake_trainer._global_step = fake_updates_done
            last_generator_checkpoint = generator_trainer._save_checkpoint()
            last_fake_checkpoint = fake_trainer._save_checkpoint()
            last_generator_saved_step = step
            last_fake_saved_update = fake_updates_done

    generator_trainer._global_step = args.steps
    fake_trainer._global_step = fake_updates_done
    if last_generator_checkpoint is None or last_generator_saved_step != args.steps:
        last_generator_checkpoint = generator_trainer._save_checkpoint()
    if last_fake_checkpoint is None or last_fake_saved_update != fake_updates_done:
        last_fake_checkpoint = fake_trainer._save_checkpoint()
    return last_generator_checkpoint, last_fake_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Initial paired-regression generator adapter.")
    parser.add_argument("--fake-checkpoint", type=Path, help="Optional fake-score adapter for resume.")
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--fake-rank", type=int, default=4)
    parser.add_argument("--generator-lr", type=float, default=1e-6)
    parser.add_argument("--fake-lr", type=float, default=1e-5)
    parser.add_argument("--paired-weight", type=float, default=1.0)
    parser.add_argument("--dm-weight", type=float, default=0.05)
    parser.add_argument("--fake-warmup-steps", type=int, default=0)
    parser.add_argument("--fake-updates", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-module", action="append", dest="target_modules")
    args = parser.parse_args()
    if args.fake_updates <= 0 or args.fake_warmup_steps < 0 or args.checkpoint_interval <= 0:
        parser.error("fake updates/checkpoint interval must be positive and fake warmup cannot be negative")
    try:
        checkpoints = train(args)
    except ValueError as exc:
        parser.error(str(exc))
    print("generator=" + str(checkpoints[0]))
    print("fake_score=" + str(checkpoints[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
