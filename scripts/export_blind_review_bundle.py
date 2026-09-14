#!/usr/bin/env python3
"""Export only anonymous media and metadata from a blind qualification run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_MEDIA_KEYS = ("A", "B", "side_by_side_muted")


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("blind review media path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("blind review media path must stay inside the bundle")
    return path


def export_bundle(source: Path, output: Path) -> int:
    """Copy the public A/B review surface without the hidden mapping or raw renders."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("blind review output must be absent or empty")
    review_path = source / "review-index.json"
    review = json.loads(review_path.read_text())
    if not isinstance(review, dict) or review.get("blinded") is not True:
        raise ValueError("review-index.json is not marked as blinded")
    cases = review.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("blind review index contains no cases")

    media: list[Path] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("blind review case must be an object")
        for key in _MEDIA_KEYS:
            relative = _safe_relative(case.get(key))
            source_path = source / relative
            if not source_path.is_file():
                raise FileNotFoundError(f"blind review media is missing: {relative}")
            media.append(relative)

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(review_path, output / review_path.name)
    metrics = source / "anonymous-metrics.json"
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
