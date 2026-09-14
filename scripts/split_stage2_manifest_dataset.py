#!/usr/bin/env python3
"""Create prompt-disjoint train/validation views of a captured manifest dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LATENT_SOURCES = (
    "stage2_video_start_latents",
    "stage2_video_intermediate_latents",
    "stage2_video_terminal_latents",
    "stage2_audio_start_latents",
    "stage2_audio_intermediate_latents",
    "stage2_audio_terminal_latents",
)


def split_dataset(manifest: Path, data_root: Path, output_root: Path) -> dict[str, int]:
    """Hard-link a complete capture into manifest-defined split directories."""
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    source_root = data_root / ".precomputed" if (data_root / ".precomputed").is_dir() else data_root
    counts: dict[str, int] = {}
    for split in {str(record["split"]) for record in records}:
        destination = output_root / split / ".precomputed"
        if destination.exists():
            raise FileExistsError(f"refusing to merge into existing split: {destination}")

    for record in records:
        split = str(record["split"])
        index = int(record["index"])
        latent_name = f"latent_{index:04d}.safetensors"
        condition_name = f"condition_{index:04d}.safetensors"
        files = [(source, latent_name) for source in LATENT_SOURCES] + [("conditions", condition_name)]
        for source, name in files:
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
    args = parser.parse_args()
    counts = split_dataset(args.manifest, args.data, args.output)
    print("created " + ", ".join(f"{split}={count}" for split, count in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
