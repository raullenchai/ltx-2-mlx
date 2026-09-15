"""Fail-closed validation for portable compressed stage-1 checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from ltx_core_mlx.loader.integrity import transformer_sha256

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class FastStage1Contract:
    """Validated schedule and compatibility data carried by a stage-1 adapter."""

    capability: str
    schedule: tuple[float, ...]
    noise_step_indices: tuple[int, ...] | None
    noise_step_spans: tuple[tuple[int, int], ...] | None
    noise_reference_sigmas: tuple[float, ...] | None
    noise_total_steps: int
    clean_final_transition: bool
    lora_rank: int
    lora_alpha: float
    base_model_id: str
    base_revision: str
    transformer_file: str
    transformer_sha256: str
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


def _required(metadata: dict[str, str], key: str) -> str:
    value = metadata.get(key, "").strip()
    if not value:
        raise ValueError(f"fast stage-1 checkpoint is missing {key!r}")
    return value


def _validate_lora_shapes(checkpoint, rank: int) -> None:
    shapes = {name: checkpoint.get_slice(name).get_shape() for name in checkpoint.keys()}  # noqa: SIM118
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefixes = {name[: -len(a_suffix)] for name in shapes if name.endswith(a_suffix)}
    prefixes.update(name[: -len(b_suffix)] for name in shapes if name.endswith(b_suffix))
    if not prefixes:
        raise ValueError("fast stage-1 checkpoint contains no LoRA tensor pairs")
    for prefix in sorted(prefixes):
        a_name, b_name = f"{prefix}{a_suffix}", f"{prefix}{b_suffix}"
        if a_name not in shapes or b_name not in shapes:
            raise ValueError(f"incomplete LoRA tensor pair for {prefix!r}")
        a_shape, b_shape = shapes[a_name], shapes[b_name]
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != rank or b_shape[1] != rank:
            raise ValueError(f"LoRA tensor pair for {prefix!r} does not match declared rank {rank}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(model_dir: Path) -> str:
    for name in ("embedded_config.json", "config.json"):
        path = model_dir / name
        if path.is_file():
            return _file_sha256(path)
    raise ValueError("fast stage-1 requires embedded_config.json or config.json")


def _package_file(model_dir: Path, value: object, key: str, *, suffix: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"fast stage-1 manifest is missing {key!r}")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != suffix:
        raise ValueError(f"fast stage-1 manifest {key!r} must name one local {suffix} file")
    path = model_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"fast stage-1 package file not found: {path}")
    return path


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


def _parse_noise_spans(value: str, *, transitions: int, total_steps: int) -> tuple[tuple[int, int], ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("fast_stage1_noise_step_spans must be valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != transitions:
        raise ValueError("fast_stage1_noise_step_spans requires one span per transition")
    if any(
        not isinstance(span, list)
        or len(span) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in span)
        for span in raw
    ):
        raise ValueError("fast_stage1_noise_step_spans must contain integer pairs")
    spans = tuple((span[0], span[1]) for span in raw)
    if spans[0][0] != 0 or any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("fast_stage1_noise_step_spans must be contiguous from zero")
    if any(not 0 <= start < end <= total_steps for start, end in spans) or spans[-1][1] != total_steps:
        raise ValueError("fast_stage1_noise_step_spans must cover the original schedule")
    return spans


def _parse_reference_sigmas(value: str, *, total_steps: int) -> tuple[float, ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("fast_stage1_noise_reference_sigmas must be valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != total_steps + 1:
        raise ValueError("fast_stage1_noise_reference_sigmas must contain the complete schedule")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        raise ValueError("fast_stage1_noise_reference_sigmas must contain only numbers")
    sigmas = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in sigmas) or not all(
        left > right for left, right in zip(sigmas, sigmas[1:])
    ):
        raise ValueError("fast_stage1_noise_reference_sigmas must decrease strictly")
    return sigmas


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
    for key in ("base_model_id", "base_revision", "transformer_sha256"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"fast stage-1 manifest is missing {key!r}")
    manifest_transformer_sha256 = manifest["transformer_sha256"].lower()
    if not _SHA256_RE.fullmatch(manifest_transformer_sha256):
        raise ValueError("fast stage-1 manifest transformer_sha256 must be a lowercase SHA-256 digest")
    actual_transformer_sha256 = transformer_sha256(transformer_path)
    if manifest_transformer_sha256 != actual_transformer_sha256:
        raise ValueError("fast stage-1 manifest transformer content digest mismatch")

    contract = read_fast_stage1_contract(
        adapter_path,
        base_model_id=manifest["base_model_id"],
        base_revision=manifest["base_revision"],
        transformer_file=transformer_path.name,
        transformer_sha256=actual_transformer_sha256,
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
    transformer_sha256: str,
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
        if capability not in (
            "ltx_stage1_compressed_v1",
            "ltx_stage1_compressed_span_v2",
            "ltx_stage1_compressed_span_v2_clean_final",
        ):
            raise ValueError(f"unsupported fast stage-1 capability {capability!r}")
        schedule = _parse_schedule(_required(metadata, "fast_stage1_schedule"))
        if int(_required(metadata, "stage1_steps")) != len(schedule) - 1:
            raise ValueError("ltx_stage1_compressed_v1 requires stage1_steps=4")
        noise_total_steps = int(_required(metadata, "fast_stage1_noise_total_steps"))
        if noise_total_steps <= 0:
            raise ValueError("fast_stage1_noise_total_steps must be positive")
        noise_step_indices = None
        noise_step_spans = None
        noise_reference_sigmas = None
        clean_final_transition = capability == "ltx_stage1_compressed_span_v2_clean_final"
        if capability == "ltx_stage1_compressed_v1":
            noise_step_indices = _parse_noise_indices(
                _required(metadata, "fast_stage1_noise_step_indices"),
                transitions=len(schedule) - 1,
                total_steps=noise_total_steps,
            )
            if _required(metadata, "stage1_sampler") != "ancestral_compressed":
                raise ValueError("fast stage-1 v1 requires the ancestral_compressed sampler")
        else:
            noise_step_spans = _parse_noise_spans(
                _required(metadata, "fast_stage1_noise_step_spans"),
                transitions=len(schedule) - 1,
                total_steps=noise_total_steps,
            )
            noise_reference_sigmas = _parse_reference_sigmas(
                _required(metadata, "fast_stage1_noise_reference_sigmas"),
                total_steps=noise_total_steps,
            )
            if any(
                schedule[index] != noise_reference_sigmas[span[0]]
                or schedule[index + 1] != noise_reference_sigmas[span[1]]
                for index, span in enumerate(noise_step_spans)
            ):
                raise ValueError("fast stage-1 noise spans do not match its coarse schedule")
            expected_sampler = (
                "ancestral_compressed_span_v2_clean_final"
                if clean_final_transition
                else "ancestral_compressed_span_v2"
            )
            if _required(metadata, "stage1_sampler") != expected_sampler:
                raise ValueError("fast stage-1 v2 requires the ancestral_compressed_span_v2 sampler")
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
        declared_transformer_sha256 = _required(metadata, "transformer_sha256").lower()
        declared_config_sha256 = _required(metadata, "transformer_config_sha256").lower()
        if not _IMMUTABLE_REVISION_RE.fullmatch(declared_revision):
            raise ValueError("base_revision must be an immutable hexadecimal revision")
        if not _SHA256_RE.fullmatch(declared_config_sha256):
            raise ValueError("transformer_config_sha256 must be a lowercase SHA-256 digest")
        if not _SHA256_RE.fullmatch(declared_transformer_sha256):
            raise ValueError("transformer_sha256 must be a lowercase SHA-256 digest")
        if declared_model_id != base_model_id:
            raise ValueError("fast stage-1 base model identifier mismatch")
        if declared_revision != base_revision.lower():
            raise ValueError("fast stage-1 base revision mismatch")
        if declared_transformer != transformer_file:
            raise ValueError("fast stage-1 transformer filename mismatch")
        if declared_transformer_sha256 != transformer_sha256.lower():
            raise ValueError("fast stage-1 transformer content digest mismatch")
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
        capability=capability,
        schedule=schedule,
        noise_step_indices=noise_step_indices,
        noise_step_spans=noise_step_spans,
        noise_reference_sigmas=noise_reference_sigmas,
        noise_total_steps=noise_total_steps,
        clean_final_transition=clean_final_transition,
        lora_rank=rank,
        lora_alpha=alpha,
        base_model_id=declared_model_id,
        base_revision=declared_revision,
        transformer_file=declared_transformer,
        transformer_sha256=declared_transformer_sha256,
        transformer_config_sha256=declared_config_sha256,
        pipeline_family=pipeline_family,
        runtime_contract_major=declared_runtime_major,
        qualification_revision=qualification_revision,
    )
