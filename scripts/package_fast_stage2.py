#!/usr/bin/env python3
"""Create the portable adapter + manifest files for a fast stage-2 model revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import load_file, save_file

_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(model_dir: Path) -> Path:
    for name in ("embedded_config.json", "config.json"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError("model has no embedded_config.json or config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    parser.add_argument("--qualification-revision", required=True)
    args = parser.parse_args()

    if not _IMMUTABLE_REVISION_RE.fullmatch(args.base_revision):
        raise ValueError("--base-revision must be an immutable hexadecimal revision")
    transformer = args.model_dir / args.transformer_file
    if not transformer.is_file():
        raise FileNotFoundError(f"base transformer not found: {transformer}")

    with safe_open(args.checkpoint, framework="numpy") as source:
        metadata = source.metadata() or {}
    start_sigma = float(metadata["stage2_sigma"])
    target_sigma = float(metadata["stage2_target_sigma"])
    if not 0 < target_sigma < start_sigma <= 1:
        raise ValueError("only progressive stage-2 checkpoints can be packaged")

    output_metadata = dict(metadata)
    output_metadata.update(
        fast_stage2_capability="ltx_stage2_transition_v1",
        fast_stage2_schedule=json.dumps([start_sigma, target_sigma, 0.0], separators=(",", ":")),
        stage2_steps="2",
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        transformer_file=args.transformer_file,
        transformer_config_sha256=_sha256(_config_path(args.model_dir)),
        pipeline_family="distilled_two_stage_ltx25",
        runtime_contract_major="1",
        qualification_revision=args.qualification_revision,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_dir / "fast-stage2.safetensors"
    save_file(load_file(args.checkpoint), adapter_path, metadata=output_metadata)
    artifact_sha256 = _sha256(adapter_path)
    manifest = {
        "schema_version": 1,
        "adapter_file": adapter_path.name,
        "adapter_sha256": artifact_sha256,
        "base_model_id": args.base_model_id,
        "base_revision": args.base_revision,
        "transformer_file": args.transformer_file,
    }
    manifest_path = args.output_dir / "fast-stage2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"adapter: {adapter_path}")
    print(f"adapter sha256: {artifact_sha256}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
