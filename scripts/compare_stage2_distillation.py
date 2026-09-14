#!/usr/bin/env python3
"""Compare paired Stage-2 student/base evaluations as diagnostic evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{label} must be finite")
    return rendered


def _percent_change(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return (value / baseline - 1.0) * 100.0


def _indexed_samples(result: dict, label: str) -> dict[int, dict]:
    raw = result.get("per_sample")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} has no per_sample results")
    indexed: dict[int, dict] = {}
    for sample in raw:
        if not isinstance(sample, dict) or isinstance(sample.get("index"), bool) or not isinstance(sample.get("index"), int):
            raise ValueError(f"{label} contains a sample without an integer index")
        index = sample["index"]
        if index in indexed:
            raise ValueError(f"{label} contains duplicate sample index {index}")
        indexed[index] = sample
    return indexed


def compare_results(student: dict, baseline: dict) -> dict:
    """Return paired latent diagnostics; decoded blind review remains the quality gate."""
    if student.get("schedule") != baseline.get("schedule"):
        raise ValueError("student and baseline schedules do not match")
    student_samples = _indexed_samples(student, "student")
    baseline_samples = _indexed_samples(baseline, "baseline")
    if student_samples.keys() != baseline_samples.keys():
        raise ValueError("student and baseline sample indices do not match")

    per_sample = []
    for index in sorted(student_samples):
        item: dict[str, object] = {"index": index}
        for modality in ("video", "audio"):
            student_mse = _finite_number(student_samples[index].get(modality, {}).get("mse"), f"student {index} {modality} MSE")
            baseline_mse = _finite_number(
                baseline_samples[index].get(modality, {}).get("mse"), f"baseline {index} {modality} MSE"
            )
            item[modality] = {
                "student_mse": student_mse,
                "baseline_mse": baseline_mse,
                "mse_change_percent": _percent_change(student_mse, baseline_mse),
            }
        per_sample.append(item)

    modalities = {}
    for modality in ("video", "audio"):
        student_mse = _finite_number(student.get("aggregate", {}).get(modality, {}).get("mse"), f"student mean {modality} MSE")
        baseline_mse = _finite_number(
            baseline.get("aggregate", {}).get(modality, {}).get("mse"), f"baseline mean {modality} MSE"
        )
        changes = [(item[modality]["mse_change_percent"], item["index"]) for item in per_sample]
        finite_changes = [(change, index) for change, index in changes if change is not None]
        worst = max(finite_changes, default=(None, None))
        modalities[modality] = {
            "student_mean_mse": student_mse,
            "baseline_mean_mse": baseline_mse,
            "mean_mse_change_percent": _percent_change(student_mse, baseline_mse),
            "regressed_sample_count": sum(change is not None and change > 0.0 for change, _ in changes),
            "worst_sample_mse_change_percent": worst[0],
            "worst_sample_index": worst[1],
        }

    return {
        "note": "Latent diagnostics only; decoded blind non-inferiority is the quality gate.",
        "schedule": student["schedule"],
        "samples": len(per_sample),
        "modalities": modalities,
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_results(json.loads(args.student.read_text()), json.loads(args.baseline.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
