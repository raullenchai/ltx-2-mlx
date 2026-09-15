#!/usr/bin/env python3
"""Export only anonymous media and metadata from a blind qualification run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_MEDIA_KEYS = ("A", "B", "side_by_side_muted")
_MEDIA_FILENAMES = {
    "A": "A.mp4",
    "B": "B.mp4",
    "side_by_side_muted": "AB-side-by-side-muted.mp4",
}


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("blind review media path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("blind review media path must stay inside the bundle")
    return path


def export_bundle(source: Path, output: Path) -> int:
    """Copy the public A/B review surface without the hidden mapping or raw renders."""
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("blind review output must be absent or empty")
    review_path = source / "review-index.json"
    review = json.loads(review_path.read_text())
    if not isinstance(review, dict) or review.get("blinded") is not True:
        raise ValueError("review-index.json is not marked as blinded")
    cases = review.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("blind review index contains no cases")

    media: list[Path] = []
    source_root = source.resolve()
    for expected_index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("blind review case must be an object")
        index = case.get("index")
        if isinstance(index, bool) or index != expected_index:
            raise ValueError("blind review case indices must be contiguous and ordered")
        for key in _MEDIA_KEYS:
            relative = _safe_relative(case.get(key))
            expected = Path(f"case-{index:02d}") / _MEDIA_FILENAMES[key]
            if relative != expected:
                raise ValueError(f"blind review {key} path must be exactly {expected}")
            source_path = source / relative
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or not source_path.resolve().is_relative_to(source_root)
            ):
                raise FileNotFoundError(f"blind review media is missing: {relative}")
            media.append(relative)

    metrics = source / "anonymous-metrics.json"
    if metrics.is_file() and (
        metrics.is_symlink() or not metrics.resolve().is_relative_to(source_root)
    ):
        raise ValueError("anonymous metrics must be a regular file inside the bundle")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(review_path, output / review_path.name)
    if metrics.is_file():
        shutil.copy2(metrics, output / metrics.name)
    for relative in media:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_bundle(args.source, args.output)
    print(f"exported {count} blinded cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
