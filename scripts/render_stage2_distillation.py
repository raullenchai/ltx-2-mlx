#!/usr/bin/env python3
"""Decode a paired teacher/student stage-2 trajectory for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_pipelines_mlx.utils._orchestration import decode_and_save_video
from ltx_pipelines_mlx.utils.blocks import AudioDecoder, VideoDecoder
from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.distillation_evaluator import (
    load_terminal_student,
    predict_terminal,
    prepare_trajectory_inputs,
    terminal_sigma_schedule,
)
from ltx_trainer_mlx.model_loader import load_transformer
from ltx_trainer_mlx.training_strategies.stage2_terminal_distill import (
    Stage2TerminalDistillConfig,
    Stage2TerminalDistillStrategy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transformer-file")
    args = parser.parse_args()

    model, metadata = load_terminal_student(
        args.model,
        args.checkpoint,
        transformer_file=args.transformer_file,
    )
    schedule = terminal_sigma_schedule(metadata)
    progressive = len(schedule) == 3
    strategy = Stage2TerminalDistillStrategy(
        Stage2TerminalDistillConfig(
            sigma=schedule[0],
            target_sigma=schedule[1],
            video_terminal_latents_dir=metadata.get(
                "stage2_video_target_latents_dir",
                "stage2_video_terminal_latents",
            ),
            audio_terminal_latents_dir=metadata.get(
                "stage2_audio_target_latents_dir",
                "stage2_audio_terminal_latents",
            ),
        )
    )
    data_sources = strategy.get_data_sources()
    if progressive:
        data_sources |= {
            "stage2_video_terminal_latents": "video_teacher_terminal",
            "stage2_audio_terminal_latents": "audio_teacher_terminal",
        }
    dataset = PrecomputedDataset(args.data, data_sources=data_sources)
    sample = dataset[args.index]
    inputs = prepare_trajectory_inputs(sample, metadata)
    assert inputs.audio is not None and inputs.audio_targets is not None
    sigma, target_sigma = schedule[:2]
    student_video, student_audio, _ = predict_terminal(model, inputs, sigma, target_sigma)
    delta = sigma - target_sigma
    teacher_video = inputs.video.latent - delta * inputs.video_targets
    teacher_audio = inputs.audio.latent - delta * inputs.audio_targets
    mx.eval(student_video, student_audio, teacher_video, teacher_audio)

    del model
    aggressive_cleanup()
    if progressive:
        base_model = load_transformer(args.model, transformer_file=args.transformer_file)
        student_video, student_audio, _ = predict_terminal(
            base_model,
            inputs,
            target_sigma,
            0.0,
            video_latent=student_video,
            audio_latent=student_audio,
        )
        video_patchifier = VideoLatentPatchifier()
        audio_patchifier = AudioPatchifier()
        teacher_video, _ = video_patchifier.patchify(
            mx.expand_dims(sample["video_teacher_terminal"]["latents"], axis=0)
        )
        teacher_audio, _ = audio_patchifier.patchify(
            mx.expand_dims(sample["audio_teacher_terminal"]["latents"], axis=0)
        )
        mx.eval(student_video, student_audio, teacher_video, teacher_audio)
        del base_model
        aggressive_cleanup()
    video_meta = sample["video_start"]
    dims = tuple(int(video_meta[key].item()) for key in ("num_frames", "height", "width"))
    frame_rate = float(video_meta["fps"].item())
    video_patchifier = VideoLatentPatchifier()
    audio_patchifier = AudioPatchifier()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_decoder = VideoDecoder(args.model)
    audio_decoder = AudioDecoder(args.model)
    for label, video, audio in (
        ("teacher", teacher_video, teacher_audio),
        ("student", student_video, student_audio),
    ):
        video_latent = video_patchifier.unpatchify(video, dims)
        audio_latent = audio_patchifier.unpatchify(audio)
        decode_and_save_video(
            video_decoder,
            audio_decoder,
            video_latent,
            audio_latent,
            str(args.output_dir / f"sample-{args.index:02d}-{label}.mp4"),
            frame_rate=frame_rate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
