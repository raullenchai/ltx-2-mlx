#!/usr/bin/env python3
"""Create prompt-disjoint views of a complete captured Stage-1 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS

STAGE1_SOURCES = tuple(
    source
    for step in range(len(DISTILLED_SIGMAS))
    for source in (f"stage1_video_step_{step:02d}", f"stage1_audio_step_{step:02d}")
) + ("stage1_conditions",)


def split_stage1_dataset(
    manifest: Path,
    data_root: Path,
    output_root: Path,
    *,
    bucket: str | None = None,
) -> dict[str, int]:
    """Hard-link complete Stage-1 captures into manifest-defined splits."""
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if bucket is not None:
        records = [record for record in records if str(record.get("bucket")) == bucket]
    if not records:
        raise ValueError("no manifest records remain after filtering")
    source_root = data_root / ".precomputed" if (data_root / ".precomputed").is_dir() else data_root
    splits = {str(record["split"]) for record in records}
    for split in splits:
        destination = output_root / split / ".precomputed"
        if destination.exists():
            raise FileExistsError(f"refusing to merge into existing split: {destination}")

    counts: dict[str, int] = {}
    for record in records:
        split = str(record["split"])
        name = f"latent_{int(record['index']):04d}.safetensors"
        for source in STAGE1_SOURCES:
            source_path = source_root / source / name
            if not source_path.is_file():
                raise FileNotFoundError(f"missing captured component: {source_path}")
            destination = output_root / split / ".precomputed" / source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, destination)
        counts[split] = counts.get(split, 0) + 1

    for split in counts:
        subset = [record for record in records if str(record["split"]) == split]
        (output_root / split / "manifest.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in subset)
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", help="Create views for only one manifest bucket.")
    args = parser.parse_args()
    counts = split_stage1_dataset(args.manifest, args.data, args.output, bucket=args.bucket)
    print("created " + ", ".join(f"{split}={count}" for split, count in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
