#!/usr/bin/env python3
"""Package the qualified exact-prefix plus learned-middle Stage-1 route."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from safetensors import safe_open

from ltx_core_mlx.loader.fast_stage1 import _config_sha256, _file_sha256
from ltx_core_mlx.loader.fast_stage1_segmented import _read_segment
from ltx_core_mlx.loader.integrity import transformer_sha256
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS

_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_LEARNED_SPAN = (3, 7)
_EXECUTION_SPANS = ((0, 1), (1, 2), (2, 3), _LEARNED_SPAN)
_SCHEDULE = tuple(DISTILLED_SIGMAS[index] for index in (0, 1, 2, 3, 7, 8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--transformer-file", default="transformer-distilled.safetensors")
    parser.add_argument("--qualification-revision", required=True)
    parser.add_argument(
        "--symlink-adapter",
        action="store_true",
        help="Diagnostic scratch mode only; a distributable package must copy the adapter",
    )
    args = parser.parse_args()

    if not _IMMUTABLE_REVISION_RE.fullmatch(args.base_revision):
        raise ValueError("--base-revision must be an immutable hexadecimal revision")
    if args.symlink_adapter and not args.qualification_revision.startswith("diagnostic-"):
        raise ValueError("symlink adapters are restricted to diagnostic qualification revisions")
    if args.output_dir.is_symlink() or (args.output_dir.exists() and any(args.output_dir.iterdir())):
        raise ValueError("exact-prefix package output must be absent or empty")
    transformer = args.model_dir / args.transformer_file
    if not transformer.is_file():
        raise FileNotFoundError(f"base transformer not found: {transformer}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"middle checkpoint not found: {args.checkpoint}")

    with safe_open(args.checkpoint, framework="numpy") as source:
        metadata = source.metadata() or {}
    if metadata.get("stage1_curriculum_adapter_mode") != "independent":
        raise ValueError("middle checkpoint must be trained independently from the clean base")
    source_digest = _file_sha256(args.checkpoint)
    raw = {
        "adapter_file": args.checkpoint.name,
        "adapter_sha256": source_digest,
        "span": list(_LEARNED_SPAN),
    }
    _read_segment(args.checkpoint.parent, raw, expected_span=_LEARNED_SPAN, allow_symlink=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "fast-stage1-middle-3-7.safetensors"
    if args.symlink_adapter:
        destination.symlink_to(args.checkpoint.resolve())
    else:
        shutil.copy2(args.checkpoint, destination)
    segment = {
        "adapter_file": destination.name,
        "adapter_sha256": source_digest,
        "span": list(_LEARNED_SPAN),
    }
    _read_segment(
        args.output_dir,
        segment,
        expected_span=_LEARNED_SPAN,
        allow_symlink=args.symlink_adapter,
    )

    manifest = {
        "schema_version": 1,
        "capability": "ltx_stage1_exact_prefix_middle_span_v1",
        "schedule": list(_SCHEDULE),
        "execution_spans": [list(span) for span in _EXECUTION_SPANS],
        "learned_spans": [list(_LEARNED_SPAN)],
        "clean_final_span": [7, 8],
        "noise_reference_sigmas": list(DISTILLED_SIGMAS),
        "segments": [segment],
        "base_model_id": args.base_model_id,
        "base_revision": args.base_revision,
        "transformer_file": args.transformer_file,
        "transformer_sha256": transformer_sha256(transformer),
        "transformer_config_sha256": _config_sha256(args.model_dir),
        "pipeline_family": "distilled_two_stage_ltx25",
        "runtime_contract_major": 1,
        "qualification_revision": args.qualification_revision,
    }
    manifest_path = args.output_dir / "fast-stage1-exact-prefix.json"
    temporary = args.output_dir / ".fast-stage1-exact-prefix.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(f"manifest: {manifest_path}")
    print(f"middle: 3->7: {source_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
