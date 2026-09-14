"""Fail-closed validation for portable fast stage-2 checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class FastStage2Contract:
    """Validated stage-2 transition contract carried by an adapter."""

    capability: str
    schedule: tuple[float, ...]
    lora_rank: int
    lora_alpha: float
    base_model_id: str
    base_revision: str
    transformer_file: str
    transformer_config_sha256: str
    runtime_contract_major: int
    qualification_revision: str


@dataclass(frozen=True)
class FastStage2Package:
    """A validated adapter and the exact base transformer it extends."""

    adapter_path: Path
    transformer_path: Path
    artifact_sha256: str
    contract: FastStage2Contract


def _required(metadata: dict[str, str], key: str) -> str:
    value = metadata.get(key, "").strip()
    if not value:
        raise ValueError(f"fast stage-2 checkpoint is missing {key!r}")
    return value


def _parse_schedule(value: str) -> tuple[float, ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("fast_stage2_schedule must be valid JSON") from exc
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("ltx_stage2_transition_v1 requires a three-sigma schedule")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        raise ValueError("fast_stage2_schedule must contain only numbers")
    schedule = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in schedule):
        raise ValueError("fast_stage2_schedule must contain finite numbers")
    if schedule[-1] != 0.0 or not all(left > right for left, right in zip(schedule, schedule[1:])):
        raise ValueError("fast_stage2_schedule must decrease strictly to zero")
    if schedule[0] > 1.0:
        raise ValueError("fast_stage2_schedule must start in (0, 1]")
    return schedule


def _validate_lora_shapes(checkpoint, rank: int) -> None:
    # Some supported safetensors releases expose ``keys()`` without making the
    # safe-open handle iterable.
    shapes = {name: checkpoint.get_slice(name).get_shape() for name in checkpoint.keys()}  # noqa: SIM118
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefixes = {name[: -len(a_suffix)] for name in shapes if name.endswith(a_suffix)}
    prefixes.update(name[: -len(b_suffix)] for name in shapes if name.endswith(b_suffix))
    if not prefixes:
        raise ValueError("fast stage-2 checkpoint contains no LoRA tensor pairs")

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
    raise ValueError("fast stage-2 requires embedded_config.json or config.json")


def _package_file(model_dir: Path, value: object, key: str, *, suffix: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"fast stage-2 manifest is missing {key!r}")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix != suffix:
        raise ValueError(f"fast stage-2 manifest {key!r} must name one local {suffix} file")
    path = model_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"fast stage-2 package file not found: {path}")
    return path


def read_fast_stage2_package(
    model_dir: str | Path,
    manifest_name: str = "fast-stage2.json",
    *,
    runtime_contract_major: int = 1,
) -> FastStage2Package:
    """Read and validate a self-contained fast-stage package from a model revision."""
    root = Path(model_dir)
    manifest_path = _package_file(root, manifest_name, "manifest_name", suffix=".json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("fast stage-2 manifest must be valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("fast stage-2 manifest requires schema_version=1")

    adapter_path = _package_file(root, manifest.get("adapter_file"), "adapter_file", suffix=".safetensors")
    transformer_path = _package_file(
        root,
        manifest.get("transformer_file"),
        "transformer_file",
        suffix=".safetensors",
    )
    artifact_sha256 = manifest.get("adapter_sha256")
    if not isinstance(artifact_sha256, str):
        raise ValueError("fast stage-2 manifest is missing 'adapter_sha256'")

    string_fields = ("base_model_id", "base_revision")
    for key in string_fields:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"fast stage-2 manifest is missing {key!r}")

    contract = read_fast_stage2_contract(
        adapter_path,
        base_model_id=manifest["base_model_id"],
        base_revision=manifest["base_revision"],
        transformer_file=transformer_path.name,
        transformer_config_sha256=_config_sha256(root),
        runtime_contract_major=runtime_contract_major,
        expected_artifact_sha256=artifact_sha256,
    )
    return FastStage2Package(
        adapter_path=adapter_path,
        transformer_path=transformer_path,
        artifact_sha256=artifact_sha256,
        contract=contract,
    )


def read_fast_stage2_contract(
    checkpoint_path: str | Path,
    *,
    base_model_id: str,
    base_revision: str,
    transformer_file: str,
    transformer_config_sha256: str,
    runtime_contract_major: int = 1,
    expected_artifact_sha256: str | None = None,
) -> FastStage2Contract:
    """Validate an adapter before it can mutate the loaded base transformer.

    Hardware identity is deliberately absent: the same artifact contract applies
    to every Apple Silicon host that can admit the base workload.
    """
    path = Path(checkpoint_path)
    if expected_artifact_sha256 is not None:
        expected_artifact_sha256 = expected_artifact_sha256.lower()
        if not _SHA256_RE.fullmatch(expected_artifact_sha256):
            raise ValueError("expected_artifact_sha256 must be a lowercase SHA-256 digest")
        if _file_sha256(path) != expected_artifact_sha256:
            raise ValueError("fast stage-2 artifact digest mismatch")

    with safe_open(path, framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
        capability = _required(metadata, "fast_stage2_capability")
        if capability != "ltx_stage2_transition_v1":
            raise ValueError(f"unsupported fast stage-2 capability {capability!r}")

        schedule = _parse_schedule(_required(metadata, "fast_stage2_schedule"))
        start_sigma = float(_required(metadata, "stage2_sigma"))
        target_sigma = float(_required(metadata, "stage2_target_sigma"))
        if not math.isclose(start_sigma, schedule[0], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stage2_sigma does not match fast_stage2_schedule")
        if not math.isclose(target_sigma, schedule[1], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stage2_target_sigma does not match fast_stage2_schedule")
        if int(_required(metadata, "stage2_steps")) != 2:
            raise ValueError("ltx_stage2_transition_v1 requires stage2_steps=2")

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
            raise ValueError("fast stage-2 base model identifier mismatch")
        if declared_revision != base_revision.lower():
            raise ValueError("fast stage-2 base revision mismatch")
        if declared_transformer != transformer_file:
            raise ValueError("fast stage-2 transformer filename mismatch")
        if declared_config_sha256 != transformer_config_sha256.lower():
            raise ValueError("fast stage-2 transformer config fingerprint mismatch")

        declared_runtime_major = int(_required(metadata, "runtime_contract_major"))
        if declared_runtime_major != runtime_contract_major:
            raise ValueError("fast stage-2 runtime contract is incompatible")
        qualification_revision = _required(metadata, "qualification_revision")
        _validate_lora_shapes(checkpoint, rank)

    return FastStage2Contract(
        capability=capability,
        schedule=schedule,
        lora_rank=rank,
        lora_alpha=alpha,
        base_model_id=declared_model_id,
        base_revision=declared_revision,
        transformer_file=declared_transformer,
        transformer_config_sha256=declared_config_sha256,
        runtime_contract_major=declared_runtime_major,
        qualification_revision=qualification_revision,
    )
