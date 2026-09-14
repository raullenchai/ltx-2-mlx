"""Fail-closed validation for portable compressed stage-1 checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from ltx_core_mlx.loader.fast_stage2 import (
    _IMMUTABLE_REVISION_RE,
    _SHA256_RE,
    _config_sha256,
    _file_sha256,
    _package_file,
    _required,
    _validate_lora_shapes,
)


@dataclass(frozen=True)
class FastStage1Contract:
    """Validated schedule and compatibility data carried by a stage-1 adapter."""

    capability: str
    schedule: tuple[float, ...]
    noise_step_indices: tuple[int, ...]
    noise_total_steps: int
    lora_rank: int
    lora_alpha: float
    base_model_id: str
    base_revision: str
    transformer_file: str
    transformer_config_sha256: str
    pipeline_family: str
    runtime_contract_major: int
    qualification_revision: str


@dataclass(frozen=True)
class FastStage1Package:
    """A validated stage-1 adapter and the exact base transformer it extends."""

    adapter_path: Path
    transformer_path: Path
    artifact_sha256: str
    contract: FastStage1Contract


def _parse_schedule(value: str) -> tuple[float, ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("fast_stage1_schedule must be valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != 5:
        raise ValueError("ltx_stage1_compressed_v1 requires a five-sigma schedule")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        raise ValueError("fast_stage1_schedule must contain only numbers")
    schedule = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in schedule):
        raise ValueError("fast_stage1_schedule must contain finite numbers")
    if schedule[0] != 1.0 or schedule[-1] != 0.0:
        raise ValueError("fast_stage1_schedule must run from one to zero")
    if not all(left > right for left, right in zip(schedule, schedule[1:])):
        raise ValueError("fast_stage1_schedule must decrease strictly")
    return schedule


def _parse_noise_indices(value: str, *, transitions: int, total_steps: int) -> tuple[int, ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("fast_stage1_noise_step_indices must be valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != transitions:
        raise ValueError("fast_stage1_noise_step_indices requires one lane per transition")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ValueError("fast_stage1_noise_step_indices must contain only integers")
    indices = tuple(raw)
    if indices[0] != 0 or any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("fast_stage1_noise_step_indices must start at zero and increase strictly")
    if any(not 0 <= index < total_steps for index in indices):
        raise ValueError("fast_stage1_noise_step_indices entries exceed the original schedule")
    return indices


def read_fast_stage1_package(
    model_dir: str | Path,
    manifest_name: str = "fast-stage1.json",
    *,
    runtime_contract_major: int = 1,
) -> FastStage1Package:
    """Read and validate a self-contained compressed stage-1 package."""
    root = Path(model_dir)
    manifest_path = _package_file(root, manifest_name, "manifest_name", suffix=".json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("fast stage-1 manifest must be valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("fast stage-1 manifest requires schema_version=1")

    adapter_path = _package_file(root, manifest.get("adapter_file"), "adapter_file", suffix=".safetensors")
    transformer_path = _package_file(root, manifest.get("transformer_file"), "transformer_file", suffix=".safetensors")
    artifact_sha256 = manifest.get("adapter_sha256")
    if not isinstance(artifact_sha256, str):
        raise ValueError("fast stage-1 manifest is missing 'adapter_sha256'")
    for key in ("base_model_id", "base_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"fast stage-1 manifest is missing {key!r}")

    contract = read_fast_stage1_contract(
        adapter_path,
        base_model_id=manifest["base_model_id"],
        base_revision=manifest["base_revision"],
        transformer_file=transformer_path.name,
        transformer_config_sha256=_config_sha256(root),
        runtime_contract_major=runtime_contract_major,
        expected_artifact_sha256=artifact_sha256,
    )
    return FastStage1Package(adapter_path, transformer_path, artifact_sha256, contract)


def read_fast_stage1_contract(
    checkpoint_path: str | Path,
    *,
    base_model_id: str,
    base_revision: str,
    transformer_file: str,
    transformer_config_sha256: str,
    runtime_contract_major: int = 1,
    expected_artifact_sha256: str | None = None,
) -> FastStage1Contract:
    """Validate a stage-1 adapter before it can mutate the base transformer."""
    path = Path(checkpoint_path)
    if expected_artifact_sha256 is not None:
        expected_artifact_sha256 = expected_artifact_sha256.lower()
        if not _SHA256_RE.fullmatch(expected_artifact_sha256):
            raise ValueError("expected_artifact_sha256 must be a lowercase SHA-256 digest")
        if _file_sha256(path) != expected_artifact_sha256:
            raise ValueError("fast stage-1 artifact digest mismatch")

    with safe_open(path, framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
        capability = _required(metadata, "fast_stage1_capability")
        if capability != "ltx_stage1_compressed_v1":
            raise ValueError(f"unsupported fast stage-1 capability {capability!r}")
        schedule = _parse_schedule(_required(metadata, "fast_stage1_schedule"))
        if int(_required(metadata, "stage1_steps")) != len(schedule) - 1:
            raise ValueError("ltx_stage1_compressed_v1 requires stage1_steps=4")
        noise_total_steps = int(_required(metadata, "fast_stage1_noise_total_steps"))
        if noise_total_steps <= 0:
            raise ValueError("fast_stage1_noise_total_steps must be positive")
        noise_step_indices = _parse_noise_indices(
            _required(metadata, "fast_stage1_noise_step_indices"),
            transitions=len(schedule) - 1,
            total_steps=noise_total_steps,
        )
        if _required(metadata, "stage1_sampler") != "ancestral_compressed":
            raise ValueError("fast stage-1 requires the ancestral_compressed sampler")
        if float(_required(metadata, "stage1_ancestral_eta")) != 1.0:
            raise ValueError("fast stage-1 requires ancestral eta=1")
        if float(_required(metadata, "stage1_ancestral_s_noise")) != 1.0:
            raise ValueError("fast stage-1 requires ancestral s_noise=1")

        rank = int(_required(metadata, "lora_rank"))
        alpha = float(_required(metadata, "lora_alpha"))
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")

        declared_model_id = _required(metadata, "base_model_id")
        declared_revision = _required(metadata, "base_revision").lower()
        declared_transformer = _required(metadata, "transformer_file")
        declared_config_sha256 = _required(metadata, "transformer_config_sha256").lower()
        if not _IMMUTABLE_REVISION_RE.fullmatch(declared_revision):
            raise ValueError("base_revision must be an immutable hexadecimal revision")
        if not _SHA256_RE.fullmatch(declared_config_sha256):
            raise ValueError("transformer_config_sha256 must be a lowercase SHA-256 digest")
        if declared_model_id != base_model_id:
            raise ValueError("fast stage-1 base model identifier mismatch")
        if declared_revision != base_revision.lower():
            raise ValueError("fast stage-1 base revision mismatch")
        if declared_transformer != transformer_file:
            raise ValueError("fast stage-1 transformer filename mismatch")
        if declared_config_sha256 != transformer_config_sha256.lower():
            raise ValueError("fast stage-1 transformer config fingerprint mismatch")

        declared_runtime_major = int(_required(metadata, "runtime_contract_major"))
        pipeline_family = _required(metadata, "pipeline_family")
        if pipeline_family != "distilled_two_stage_ltx25":
            raise ValueError(f"unsupported fast stage-1 pipeline family {pipeline_family!r}")
        if declared_runtime_major != runtime_contract_major:
            raise ValueError("fast stage-1 runtime contract is incompatible")
        qualification_revision = _required(metadata, "qualification_revision")
        _validate_lora_shapes(checkpoint, rank)

    return FastStage1Contract(
        capability,
        schedule,
        noise_step_indices,
        noise_total_steps,
        rank,
        alpha,
        declared_model_id,
        declared_revision,
        declared_transformer,
        declared_config_sha256,
        pipeline_family,
        declared_runtime_major,
        qualification_revision,
    )
