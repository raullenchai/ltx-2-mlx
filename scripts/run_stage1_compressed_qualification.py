#!/usr/bin/env python3
"""Gate, train, and evaluate the full compressed Stage-1 prototype."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def pilot_gate(
    result: dict,
    *,
    minimum_improvement_percent: float,
    max_sample_regression_percent: float,
) -> list[str]:
    """Return reasons a first-transition pilot must not advance."""
    reasons = []
    for modality in ("video", "audio"):
        change = result.get(f"{modality}_mse_change_percent")
        if not isinstance(change, (int, float)) or change > -minimum_improvement_percent:
            reasons.append(f"{modality} mean MSE improvement is below {minimum_improvement_percent:g}%")

    student_samples = result.get("student", {}).get("samples", [])
    baseline_samples = result.get("baseline", {}).get("samples", [])
    if not isinstance(student_samples, list) or not isinstance(baseline_samples, list) or not student_samples:
        reasons.append("pilot has incomplete paired samples")
        return reasons
    indexed: list[dict[int, dict]] = []
    for label, samples in (("student", student_samples), ("baseline", baseline_samples)):
        by_index = {}
        for sample in samples:
            index = sample.get("index") if isinstance(sample, dict) else None
            if isinstance(index, bool) or not isinstance(index, int) or index in by_index:
                reasons.append(f"pilot has invalid or duplicate {label} sample index")
                return reasons
            by_index[index] = sample
        indexed.append(by_index)
    student_by_index, baseline_by_index = indexed
    if student_by_index.keys() != baseline_by_index.keys():
        reasons.append("pilot has incomplete paired samples")
        return reasons
    allowed_ratio = 1.0 + max_sample_regression_percent / 100.0
    for index in sorted(student_by_index):
        student, baseline = student_by_index[index], baseline_by_index[index]
        for modality in ("video", "audio"):
            student_mse = student.get(modality, {}).get("mse")
            baseline_mse = baseline.get(modality, {}).get("mse")
            if not isinstance(student_mse, (int, float)) or not isinstance(baseline_mse, (int, float)):
                reasons.append(f"sample {index} has incomplete {modality} MSE")
            elif student_mse > baseline_mse * allowed_ratio:
                reasons.append(f"sample {index} {modality} MSE regresses more than {max_sample_regression_percent:g}%")
    return reasons


def _write_status(root: Path, value: str) -> None:
    temporary = root / ".status.tmp"
    temporary.write_text(value + "\n")
    temporary.replace(root / "status")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    parser.add_argument("--noise-coupling", choices=("lane", "span-v2"), default="lane")
    parser.add_argument("--minimum-improvement-percent", type=float, default=10.0)
    parser.add_argument("--max-sample-regression-percent", type=float, default=5.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        result = json.loads(args.pilot_result.read_text())
        reasons = pilot_gate(
            result,
            minimum_improvement_percent=args.minimum_improvement_percent,
            max_sample_regression_percent=args.max_sample_regression_percent,
        )
        if reasons:
            (args.output_root / "pilot-gate.json").write_text(
                json.dumps({"passed": False, "reasons": reasons}, indent=2) + "\n"
            )
            _write_status(args.output_root, "blocked-pilot-quality-gate")
            return 2
        (args.output_root / "pilot-gate.json").write_text(json.dumps({"passed": True, "reasons": []}, indent=2) + "\n")

        curriculum = Path(__file__).with_name("train_stage1_compressed_curriculum.py")
        _write_status(args.output_root, "curriculum-started")
        with (args.output_root / "curriculum.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(curriculum),
                    "--model",
                    str(args.model),
                    "--data",
                    str(args.training_data),
                    "--output-root",
                    str(args.output_root / "curriculum"),
                    "--transformer-file",
                    args.transformer_file,
                    "--noise-coupling",
                    args.noise_coupling,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        checkpoint = args.output_root / "curriculum/replay-7-8/checkpoints/lora_weights_step_00024.safetensors"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"compressed curriculum did not produce {checkpoint}")

        evaluator = Path(__file__).with_name("evaluate_stage1_compressed_curriculum.py")
        _write_status(args.output_root, "evaluation-started")
        with (args.output_root / "evaluation.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(evaluator),
                    "--model",
                    str(args.model),
                    "--checkpoint",
                    str(checkpoint),
                    "--data",
                    str(args.validation_data),
                    "--transformer-file",
                    args.transformer_file,
                    "--output",
                    str(args.output_root / "evaluation.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except Exception:
        _write_status(args.output_root, "qualification-failed")
        raise
    _write_status(args.output_root, "evaluation-complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
