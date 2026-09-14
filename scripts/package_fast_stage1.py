#!/usr/bin/env python3
"""Create a portable adapter + manifest for compressed LTX-2.5 stage 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from ltx_core_mlx.loader.integrity import transformer_sha256

_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_SCHEDULE = [1.0, 0.98125, 0.909375, 0.421875, 0.0]
_NOISE_STEP_INDICES = [0, 3, 5, 7]


def _validate_source_checkpoint(metadata: dict[str, str]) -> None:
    expected = {
        "distillation": "stage1_transition",
        "stage1_sigma": "0.421875",
        "stage1_target_sigma": "0.0",
        "stage1_video_start_latents_dir": "stage1_video_step_07",
        "stage1_video_target_latents_dir": "stage1_video_step_08",
        "stage1_audio_start_latents_dir": "stage1_audio_step_07",
        "stage1_audio_target_latents_dir": "stage1_audio_step_08",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint is not the final compressed Stage-1 curriculum artifact: {key}")
    if "stage1_noise_step_index" in metadata or metadata.get("stage1_sampler") == "ancestral":
        raise ValueError("final compressed Stage-1 checkpoint must not declare ancestral noise")


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
    _validate_source_checkpoint(metadata)
    if int(metadata.get("lora_rank", "0")) <= 0 or float(metadata.get("lora_alpha", "0")) <= 0:
        raise ValueError("checkpoint must declare a positive LoRA rank and alpha")

    output_metadata = dict(metadata)
    output_metadata.update(
        fast_stage1_capability="ltx_stage1_compressed_v1",
        fast_stage1_schedule=json.dumps(_SCHEDULE, separators=(",", ":")),
        fast_stage1_noise_step_indices=json.dumps(_NOISE_STEP_INDICES, separators=(",", ":")),
        fast_stage1_noise_total_steps="8",
        stage1_steps="4",
        stage1_sampler="ancestral_compressed",
        stage1_ancestral_eta="1.0",
        stage1_ancestral_s_noise="1.0",
        base_model_id=args.base_model_id,
        base_revision=args.base_revision,
        transformer_file=args.transformer_file,
        transformer_sha256=transformer_sha256(transformer),
        transformer_config_sha256=_sha256(_config_path(args.model_dir)),
        pipeline_family="distilled_two_stage_ltx25",
        runtime_contract_major="1",
        qualification_revision=args.qualification_revision,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_dir / "fast-stage1.safetensors"
    save_file(load_file(args.checkpoint), adapter_path, metadata=output_metadata)
    artifact_sha256 = _sha256(adapter_path)
    manifest = {
        "schema_version": 1,
        "adapter_file": adapter_path.name,
        "adapter_sha256": artifact_sha256,
        "base_model_id": args.base_model_id,
        "base_revision": args.base_revision,
        "transformer_file": args.transformer_file,
        "transformer_sha256": output_metadata["transformer_sha256"],
    }
    manifest_path = args.output_dir / "fast-stage1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"adapter: {adapter_path}")
    print(f"adapter sha256: {artifact_sha256}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
