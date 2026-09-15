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
_PROFILES = {
    1: {
        "capability": "ltx_stage1_exact_prefix_middle_span_v1",
        "learned_spans": ((3, 7),),
        "execution_spans": ((0, 1), (1, 2), (2, 3), (3, 7)),
        "schedule_indices": (0, 1, 2, 3, 7, 8),
    },
    2: {
        "capability": "ltx_stage1_exact_high_noise_two_middle_spans_v1",
        "learned_spans": ((3, 5), (5, 7)),
        "execution_spans": ((0, 1), (1, 2), (2, 3), (3, 5), (5, 7)),
        "schedule_indices": (0, 1, 2, 3, 5, 7, 8),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Independent middle adapter; provide 3-7 once or 3-5 then 5-7 twice",
    )
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
    if len(args.checkpoint) not in _PROFILES:
        raise ValueError("exact-prefix package requires one 3-7 checkpoint or two 3-5/5-7 checkpoints")
    profile = _PROFILES[len(args.checkpoint)]
    learned_spans = profile["learned_spans"]

    if not _IMMUTABLE_REVISION_RE.fullmatch(args.base_revision):
        raise ValueError("--base-revision must be an immutable hexadecimal revision")
    if args.symlink_adapter and not args.qualification_revision.startswith("diagnostic-"):
        raise ValueError("symlink adapters are restricted to diagnostic qualification revisions")
    if args.output_dir.is_symlink() or (args.output_dir.exists() and any(args.output_dir.iterdir())):
        raise ValueError("exact-prefix package output must be absent or empty")
    transformer = args.model_dir / args.transformer_file
    if not transformer.is_file():
        raise FileNotFoundError(f"base transformer not found: {transformer}")
    sources = []
    for checkpoint, span in zip(args.checkpoint, learned_spans, strict=True):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"middle checkpoint not found: {checkpoint}")
        with safe_open(checkpoint, framework="numpy") as source:
            metadata = source.metadata() or {}
        if metadata.get("stage1_curriculum_adapter_mode") != "independent":
            raise ValueError("middle checkpoint must be trained independently from the clean base")
        source_digest = _file_sha256(checkpoint)
        raw = {
            "adapter_file": checkpoint.name,
            "adapter_sha256": source_digest,
            "span": list(span),
        }
        _read_segment(checkpoint.parent, raw, expected_span=span, allow_symlink=True)
        sources.append((checkpoint, span, source_digest))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    for checkpoint, (start, end), source_digest in sources:
        destination = args.output_dir / f"fast-stage1-middle-{start}-{end}.safetensors"
        if args.symlink_adapter:
            destination.symlink_to(checkpoint.resolve())
        else:
            shutil.copy2(checkpoint, destination)
        segment = {
            "adapter_file": destination.name,
            "adapter_sha256": source_digest,
            "span": [start, end],
        }
        _read_segment(
            args.output_dir,
            segment,
            expected_span=(start, end),
            allow_symlink=args.symlink_adapter,
        )
        segments.append(segment)

    manifest = {
        "schema_version": 1,
        "capability": profile["capability"],
        "schedule": [DISTILLED_SIGMAS[index] for index in profile["schedule_indices"]],
        "execution_spans": [list(span) for span in profile["execution_spans"]],
        "learned_spans": [list(span) for span in learned_spans],
        "clean_final_span": [7, 8],
        "noise_reference_sigmas": list(DISTILLED_SIGMAS),
        "segments": segments,
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
    for segment in segments:
        print(f"middle: {segment['span'][0]}->{segment['span'][1]}: {segment['adapter_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
