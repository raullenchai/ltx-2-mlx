#!/usr/bin/env python3
"""Render a resumable blinded teacher/student stage-2 qualification suite."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from safetensors import safe_open

from ltx_trainer_mlx.datasets import PrecomputedDataset
from ltx_trainer_mlx.training_strategies.stage2_terminal_distill import (
    Stage2TerminalDistillConfig,
    Stage2TerminalDistillStrategy,
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _review_case(start: Path, index: int, output_dir: Path) -> dict[str, object]:
    with safe_open(start, framework="numpy") as checkpoint:
        prompt = (checkpoint.metadata() or {}).get("prompt", "")
    case = output_dir / f"case-{index:02d}"
    return {
        "index": index,
        "prompt": prompt,
        "A": str((case / "A.mp4").relative_to(output_dir)),
        "B": str((case / "B.mp4").relative_to(output_dir)),
        "side_by_side_muted": str((case / "AB-side-by-side-muted.mp4").relative_to(output_dir)),
        "review": {
            "preferred": None,
            "perceptible_difference": None,
            "detail_motion_sync_notes": "",
        },
    }


def _qualification_starts(data: Path, metadata: dict[str, str]) -> list[Path]:
    target_sigma = float(metadata.get("stage2_target_sigma", "0"))
    strategy = Stage2TerminalDistillStrategy(
        Stage2TerminalDistillConfig(
            sigma=float(metadata["stage2_sigma"]),
            target_sigma=target_sigma,
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
    if target_sigma > 0:
        data_sources |= {
            "stage2_video_terminal_latents": "video_teacher_terminal",
            "stage2_audio_terminal_latents": "audio_teacher_terminal",
        }
    dataset = PrecomputedDataset(str(data), data_sources=data_sources)
    source_dir = dataset.source_paths[strategy.config.video_start_latents_dir]
    return [source_dir / relative for relative in dataset.sample_files["video_start"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transformer-file")
    parser.add_argument("--mapping-seed", type=int, default=20260914)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with safe_open(args.checkpoint, framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
    starts = _qualification_starts(args.data, metadata)
    if args.limit is not None:
        starts = starts[: args.limit]
    if not starts:
        raise ValueError("qualification dataset contains no stage-2 trajectories")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(args.mapping_seed)
    mapping: dict[str, dict[str, str]] = {}
    renderer = Path(__file__).with_name("render_stage2_distillation.py")
    review_cases = []

    for progress, start in enumerate(starts, start=1):
        index = progress - 1
        case = args.output_dir / f"case-{index:02d}"
        case.mkdir(parents=True, exist_ok=True)
        side_by_side = case / "AB-side-by-side-muted.mp4"
        teacher = case / f"sample-{index:02d}-teacher.mp4"
        student = case / f"sample-{index:02d}-student.mp4"
        if not teacher.is_file() or not student.is_file():
            command = [
                sys.executable,
                str(renderer),
                "--model",
                args.model,
                "--checkpoint",
                args.checkpoint,
                "--data",
                str(args.data),
                "--index",
                str(index),
                "--output-dir",
                str(case),
            ]
            if args.transformer_file:
                command.extend(["--transformer-file", args.transformer_file])
            _run(command)

        student_is_a = bool(randomizer.getrandbits(1))
        a_source, b_source = (student, teacher) if student_is_a else (teacher, student)
        mapping[str(index)] = {
            "A": "student" if student_is_a else "teacher",
            "B": "teacher" if student_is_a else "student",
        }
        shutil.copy2(a_source, case / "A.mp4")
        shutil.copy2(b_source, case / "B.mp4")
        if not side_by_side.is_file():
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(case / "A.mp4"),
                    "-i",
                    str(case / "B.mp4"),
                    "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2[v]",
                    "-map",
                    "[v]",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    str(side_by_side),
                ]
            )
        print(f"rendered blind case {progress}/{len(starts)} (index {index})", flush=True)
        review_cases.append(_review_case(start, index, args.output_dir))

    (args.output_dir / ".blind-mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    review_index = {
        "blinded": True,
        "instructions": "Review A and B independently with audio, then the muted side-by-side. Do not open .blind-mapping.json before recording judgments.",
        "cases": review_cases,
    }
    (args.output_dir / "review-index.json").write_text(json.dumps(review_index, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
