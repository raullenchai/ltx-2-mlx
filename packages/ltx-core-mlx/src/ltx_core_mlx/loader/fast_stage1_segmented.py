"""Fail-closed loader for segmented compressed LTX-2.5 Stage-1 packages."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from ltx_core_mlx.loader.fast_stage1 import (
    _IMMUTABLE_REVISION_RE,
    _SHA256_RE,
    _config_sha256,
    _file_sha256,
    _package_file,
    _required,
    _validate_lora_shapes,
)
from ltx_core_mlx.loader.integrity import transformer_sha256

_SCHEDULE = (1.0, 0.98125, 0.909375, 0.421875, 0.0)
_REFERENCE_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
_LEARNED_SPANS = ((0, 3), (3, 5), (5, 7))
_CLEAN_FINAL_SPAN = (7, 8)


@dataclass(frozen=True)
class FastStage1Segment:
    adapter_path: Path
    artifact_sha256: str
    start_index: int
    end_index: int
    sigma: float
    target_sigma: float
    lora_rank: int
    lora_alpha: float


@dataclass(frozen=True)
class FastStage1SegmentedPackage:
    segments: tuple[FastStage1Segment, ...]
    transformer_path: Path
    schedule: tuple[float, ...]
    noise_reference_sigmas: tuple[float, ...]
    noise_total_steps: int
    base_model_id: str
    base_revision: str
    transformer_sha256: str
    transformer_config_sha256: str
    pipeline_family: str
    runtime_contract_major: int
    qualification_revision: str


def _matches_exact_sequence(value: object, expected: tuple) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    for actual, wanted in zip(value, expected, strict=True):
        if isinstance(wanted, tuple):
            if not _matches_exact_sequence(actual, wanted):
                return False
        elif isinstance(wanted, float):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or float(actual) != wanted:
                return False
        elif type(actual) is not type(wanted) or actual != wanted:
            return False
    return True


def _exact_sequence(value: object, expected: tuple, label: str) -> None:
    if not _matches_exact_sequence(value, expected):
        raise ValueError(f"segmented fast stage-1 {label} does not match the qualified contract")


def _read_segment(
    root: Path,
    raw: object,
    *,
    expected_span: tuple[int, int],
    allow_symlink: bool = False,
) -> FastStage1Segment:
    if not isinstance(raw, dict):
        raise ValueError("segmented fast stage-1 segment must be an object")
    _exact_sequence(raw.get("span"), expected_span, "segment span")
    adapter_path = _package_file(root, raw.get("adapter_file"), "adapter_file", suffix=".safetensors")
    if adapter_path.is_symlink() and not allow_symlink:
        raise ValueError("segmented fast stage-1 adapter symlinks require a diagnostic qualification")
    digest = raw.get("adapter_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("segmented fast stage-1 adapter digest must be lowercase SHA-256")
    if _file_sha256(adapter_path) != digest:
        raise ValueError("segmented fast stage-1 adapter digest mismatch")

    start, end = expected_span
    with safe_open(adapter_path, framework="numpy") as checkpoint:
        metadata = checkpoint.metadata() or {}
        expected = {
            "distillation": "stage1_transition",
            "stage1_sigma": str(_REFERENCE_SIGMAS[start]),
            "stage1_target_sigma": str(_REFERENCE_SIGMAS[end]),
            "stage1_video_start_latents_dir": f"stage1_video_step_{start:02d}",
            "stage1_video_target_latents_dir": f"stage1_video_step_{end:02d}",
            "stage1_audio_start_latents_dir": f"stage1_audio_step_{start:02d}",
            "stage1_audio_target_latents_dir": f"stage1_audio_step_{end:02d}",
            "stage1_curriculum_noise_coupling": "span-v2",
            "stage1_sampler": "ancestral_span_v2",
            "stage1_noise_step_index": str(start),
            "stage1_noise_step_end_index": str(end),
            "stage1_noise_total_steps": "8",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"segmented fast stage-1 checkpoint has incompatible {key}")
        try:
            reference_sigmas = tuple(
                float(item) for item in json.loads(_required(metadata, "stage1_noise_reference_sigmas"))
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("segmented fast stage-1 checkpoint has invalid reference sigmas") from exc
        if reference_sigmas != _REFERENCE_SIGMAS:
            raise ValueError("segmented fast stage-1 checkpoint has incompatible reference sigmas")
        rank = int(_required(metadata, "lora_rank"))
        alpha = float(_required(metadata, "lora_alpha"))
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("segmented fast stage-1 LoRA scale must be positive")
        _validate_lora_shapes(checkpoint, rank)
    return FastStage1Segment(
        adapter_path=adapter_path,
        artifact_sha256=digest,
        start_index=start,
        end_index=end,
        sigma=_REFERENCE_SIGMAS[start],
        target_sigma=_REFERENCE_SIGMAS[end],
        lora_rank=rank,
        lora_alpha=alpha,
    )


def read_fast_stage1_segmented_package(
    model_dir: str | Path,
    manifest_name: str = "fast-stage1-segmented.json",
    *,
    runtime_contract_major: int = 1,
) -> FastStage1SegmentedPackage:
    """Read three bound student segments followed by one clean-base transition."""
    root = Path(model_dir)
    manifest_path = _package_file(root, manifest_name, "manifest_name", suffix=".json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("segmented fast stage-1 manifest must be valid JSON") from exc
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("segmented fast stage-1 manifest requires schema_version=1")
    if manifest.get("capability") != "ltx_stage1_segmented_span_v1":
        raise ValueError("unsupported segmented fast stage-1 capability")
    _exact_sequence(manifest.get("schedule"), _SCHEDULE, "schedule")
    _exact_sequence(manifest.get("noise_reference_sigmas"), _REFERENCE_SIGMAS, "reference sigmas")
    _exact_sequence(manifest.get("learned_spans"), _LEARNED_SPANS, "learned spans")
    _exact_sequence(manifest.get("clean_final_span"), _CLEAN_FINAL_SPAN, "clean final span")

    base_model_id = manifest.get("base_model_id")
    base_revision = manifest.get("base_revision")
    declared_transformer_sha = manifest.get("transformer_sha256")
    declared_config_sha = manifest.get("transformer_config_sha256")
    if not isinstance(base_model_id, str) or not base_model_id:
        raise ValueError("segmented fast stage-1 manifest is missing base_model_id")
    if not isinstance(base_revision, str) or not _IMMUTABLE_REVISION_RE.fullmatch(base_revision):
        raise ValueError("segmented fast stage-1 base revision must be immutable")
    if not isinstance(declared_transformer_sha, str) or not _SHA256_RE.fullmatch(declared_transformer_sha):
        raise ValueError("segmented fast stage-1 transformer digest must be lowercase SHA-256")
    if not isinstance(declared_config_sha, str) or not _SHA256_RE.fullmatch(declared_config_sha):
        raise ValueError("segmented fast stage-1 config digest must be lowercase SHA-256")
    transformer_path = _package_file(
        root,
        manifest.get("transformer_file"),
        "transformer_file",
        suffix=".safetensors",
    )
    if transformer_sha256(transformer_path) != declared_transformer_sha:
        raise ValueError("segmented fast stage-1 transformer content digest mismatch")
    if _config_sha256(root) != declared_config_sha:
        raise ValueError("segmented fast stage-1 transformer config fingerprint mismatch")

    pipeline_family = manifest.get("pipeline_family")
    declared_runtime = manifest.get("runtime_contract_major")
    qualification_revision = manifest.get("qualification_revision")
    if pipeline_family != "distilled_two_stage_ltx25":
        raise ValueError("unsupported segmented fast stage-1 pipeline family")
    if type(declared_runtime) is not int or declared_runtime != runtime_contract_major:
        raise ValueError("segmented fast stage-1 runtime contract is incompatible")
    if not isinstance(qualification_revision, str) or not qualification_revision:
        raise ValueError("segmented fast stage-1 qualification revision is missing")
    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(_LEARNED_SPANS):
        raise ValueError("segmented fast stage-1 requires three adapters")
    segments = tuple(
        _read_segment(
            root,
            raw,
            expected_span=span,
            allow_symlink=qualification_revision.startswith("diagnostic-"),
        )
        for raw, span in zip(raw_segments, _LEARNED_SPANS, strict=True)
    )
    return FastStage1SegmentedPackage(
        segments=segments,
        transformer_path=transformer_path,
        schedule=_SCHEDULE,
        noise_reference_sigmas=_REFERENCE_SIGMAS,
        noise_total_steps=8,
        base_model_id=base_model_id,
        base_revision=base_revision,
        transformer_sha256=declared_transformer_sha,
        transformer_config_sha256=declared_config_sha,
        pipeline_family=pipeline_family,
        runtime_contract_major=declared_runtime,
        qualification_revision=qualification_revision,
    )


__all__ = [
    "FastStage1Segment",
    "FastStage1SegmentedPackage",
    "read_fast_stage1_segmented_package",
]
