#!/usr/bin/env python3
"""Benchmark and blind-review standard 8+3 against combined fast 4+1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CasePlan:
    index: int
    prompt: str
    seed: int
    fast_is_a: bool
    fast_first: bool


_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")


def _read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines()]
    prompts = [prompt for prompt in prompts if prompt and not prompt.startswith("#")]
    if not prompts:
        raise ValueError("prompt file contains no prompts")
    return prompts


def _case_plans(prompts: list[str], *, seed: int, mapping_seed: int) -> list[CasePlan]:
    randomizer = random.Random(mapping_seed)
    return [
        CasePlan(
            index=index,
            prompt=prompt,
            seed=seed + index,
            fast_is_a=bool(randomizer.getrandbits(1)),
            fast_first=bool(randomizer.getrandbits(1)),
        )
        for index, prompt in enumerate(prompts)
    ]


def _generation_command(
    *,
    model: str,
    output: Path,
    plan: CasePlan,
    width: int,
    height: int,
    frames: int,
    frame_rate: int,
    fast_stage1_manifest: str | None = None,
    fast_stage1_segmented_manifest: str | None = None,
    fast_stage2_manifest: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ltx_pipelines_mlx.cli",
        "generate",
        "--model",
        model,
        "--distilled",
        "--low-ram",
        "--quiet",
        "--prompt",
        plan.prompt,
        "--height",
        str(height),
        "--width",
        str(width),
        "--frames",
        str(frames),
        "--frame-rate",
        str(frame_rate),
        "--seed",
        str(plan.seed),
        "--output",
        str(output),
    ]
    if fast_stage1_manifest is not None and fast_stage1_segmented_manifest is not None:
        raise ValueError("choose either a shared or segmented fast stage-1 manifest")
    selected_stage1 = fast_stage1_segmented_manifest or fast_stage1_manifest
    if (selected_stage1 is None) != (fast_stage2_manifest is None):
        raise ValueError("combined fast qualification requires both manifests")
    if fast_stage1_manifest is not None:
        command.extend(["--fast-stage1-manifest", fast_stage1_manifest])
    if fast_stage1_segmented_manifest is not None:
        command.extend(["--fast-stage1-segmented-manifest", fast_stage1_segmented_manifest])
    if selected_stage1 is not None:
        command.extend(["--fast-stage2-manifest", fast_stage2_manifest])
    return command


def _run_timed(command: list[str]) -> float:
    started = time.monotonic()
    subprocess.run(command, check=True)
    return time.monotonic() - started


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _environment() -> dict[str, object]:
    memory = _sysctl("hw.memsize")
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = None
    return {
        "platform": platform.platform(),
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "memory_bytes": int(memory) if memory is not None else None,
        "python": platform.python_version(),
        "mlx": mlx_version,
    }


def _review_case(plan: CasePlan, output_dir: Path) -> dict[str, object]:
    case = output_dir / f"case-{plan.index:02d}"
    return {
        "index": plan.index,
        "prompt": plan.prompt,
        "seed": plan.seed,
        "A": str((case / "A.mp4").relative_to(output_dir)),
        "B": str((case / "B.mp4").relative_to(output_dir)),
        "side_by_side_muted": str((case / "AB-side-by-side-muted.mp4").relative_to(output_dir)),
        "review": {
            "preferred": None,
            "perceptible_difference": None,
            "detail_motion_sync_notes": "",
        },
    }


def _completed_timing(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"standard_seconds", "fast_seconds"}:
        raise ValueError(f"invalid timing checkpoint: {path}")
    return {key: float(seconds) for key, seconds in value.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    stage1 = parser.add_mutually_exclusive_group(required=True)
    stage1.add_argument("--fast-stage1-manifest")
    stage1.add_argument("--fast-stage1-segmented-manifest")
    parser.add_argument("--fast-stage2-manifest", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=241)
    parser.add_argument("--frame-rate", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping-seed", type=int, default=20260914)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    if not _IMMUTABLE_REVISION_RE.fullmatch(args.runtime_revision):
        raise ValueError("--runtime-revision must be an immutable hexadecimal revision")

    prompts = _read_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    plans = _case_plans(prompts, seed=args.seed, mapping_seed=args.mapping_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict[str, object]] = {}
    review_cases = []
    benchmark_cases = []

    for progress, plan in enumerate(plans, start=1):
        case = args.output_dir / f"case-{plan.index:02d}"
        case.mkdir(parents=True, exist_ok=True)
        standard = case / f"sample-{plan.index:02d}-standard.mp4"
        fast = case / f"sample-{plan.index:02d}-fast.mp4"
        timing_path = case / ".timing.json"
        timing = _completed_timing(timing_path)
        if timing is None:
            outputs = {"standard": standard, "fast": fast}
            order = ("fast", "standard") if plan.fast_first else ("standard", "fast")
            measured: dict[str, float] = {}
            for variant in order:
                output = outputs[variant]
                command = _generation_command(
                    model=args.model,
                    output=output,
                    plan=plan,
                    width=args.width,
                    height=args.height,
                    frames=args.frames,
                    frame_rate=args.frame_rate,
                    fast_stage1_manifest=(args.fast_stage1_manifest if variant == "fast" else None),
                    fast_stage1_segmented_manifest=(args.fast_stage1_segmented_manifest if variant == "fast" else None),
                    fast_stage2_manifest=(args.fast_stage2_manifest if variant == "fast" else None),
                )
                measured[f"{variant}_seconds"] = _run_timed(command)
            timing_path.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
            timing = measured
        if not standard.is_file() or not fast.is_file():
            raise FileNotFoundError(f"case {plan.index} is missing a completed render")

        a_source, b_source = (fast, standard) if plan.fast_is_a else (standard, fast)
        shutil.copy2(a_source, case / "A.mp4")
        shutil.copy2(b_source, case / "B.mp4")
        side_by_side = case / "AB-side-by-side-muted.mp4"
        if not side_by_side.is_file():
            subprocess.run(
                [
                    args.ffmpeg,
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
                ],
                check=True,
            )

        speedup = timing["standard_seconds"] / timing["fast_seconds"]
        mapping[str(plan.index)] = {
            "A": "fast" if plan.fast_is_a else "standard",
            "B": "standard" if plan.fast_is_a else "fast",
            "run_order": ["fast", "standard"] if plan.fast_first else ["standard", "fast"],
        }
        benchmark_cases.append(
            {
                "index": plan.index,
                "seed": plan.seed,
                **timing,
                "speedup": speedup,
                "standard_sha256": _sha256(standard),
                "fast_sha256": _sha256(fast),
            }
        )
        review_cases.append(_review_case(plan, args.output_dir))
        print(f"rendered blind case {progress}/{len(plans)}: {speedup:.3f}x", flush=True)

    mean_standard = sum(case["standard_seconds"] for case in benchmark_cases) / len(benchmark_cases)
    mean_fast = sum(case["fast_seconds"] for case in benchmark_cases) / len(benchmark_cases)
    benchmark = {
        "workload": {
            "model": args.model,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "frame_rate": args.frame_rate,
            "standard_schedule": "8+3",
            "fast_schedule": "4+1",
            "low_ram": True,
        },
        "runtime_revision": args.runtime_revision,
        "environment": _environment(),
        "cases": benchmark_cases,
        "mean_standard_seconds": mean_standard,
        "mean_fast_seconds": mean_fast,
        "aggregate_speedup": mean_standard / mean_fast,
    }
    (args.output_dir / ".blind-mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    (args.output_dir / ".benchmark-results.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n"
    )
    review_index = {
        "blinded": True,
        "instructions": (
            "Review A and B independently with audio, then the muted side-by-side. "
            "Do not open dotfiles before recording judgments."
        ),
        "cases": review_cases,
    }
    (args.output_dir / "review-index.json").write_text(json.dumps(review_index, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
