from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_core_mlx.loader.fast_stage1_segmented import read_fast_stage1_segmented_package

_REVISION = "a" * 40
_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
_SPANS = ((0, 3), (3, 5), (5, 7))


def _write_package(tmp_path):
    transformer = tmp_path / "transformer-distilled.safetensors"
    transformer.write_bytes(b"base")
    config = tmp_path / "embedded_config.json"
    config.write_text('{"transformer":{"model_version":"2.5.0"}}')
    segments = []
    for start, end in _SPANS:
        path = tmp_path / f"segment-{start}-{end}.safetensors"
        save_file(
            {
                "block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
                "block.to_q.lora_B.weight": np.zeros((8, 2), dtype=np.float32),
            },
            path,
            metadata={
                "distillation": "stage1_transition",
                "stage1_sigma": str(_SIGMAS[start]),
                "stage1_target_sigma": str(_SIGMAS[end]),
                "stage1_video_start_latents_dir": f"stage1_video_step_{start:02d}",
                "stage1_video_target_latents_dir": f"stage1_video_step_{end:02d}",
                "stage1_audio_start_latents_dir": f"stage1_audio_step_{start:02d}",
                "stage1_audio_target_latents_dir": f"stage1_audio_step_{end:02d}",
                "stage1_curriculum_noise_coupling": "span-v2",
                "stage1_sampler": "ancestral_span_v2",
                "stage1_noise_step_index": str(start),
                "stage1_noise_step_end_index": str(end),
                "stage1_noise_total_steps": "8",
                "stage1_noise_reference_sigmas": json.dumps(_SIGMAS),
                "lora_rank": "2",
                "lora_alpha": "2",
            },
        )
        segments.append(
            {
                "adapter_file": path.name,
                "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "span": [start, end],
            }
        )
    manifest = {
        "schema_version": 1,
        "capability": "ltx_stage1_segmented_span_v1",
        "schedule": [1.0, 0.98125, 0.909375, 0.421875, 0.0],
        "learned_spans": [list(span) for span in _SPANS],
        "clean_final_span": [7, 8],
        "noise_reference_sigmas": list(_SIGMAS),
        "segments": segments,
        "base_model_id": "example/ltx-2.5-mlx-q8",
        "base_revision": _REVISION,
        "transformer_file": transformer.name,
        "transformer_sha256": hashlib.sha256(transformer.read_bytes()).hexdigest(),
        "transformer_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "pipeline_family": "distilled_two_stage_ltx25",
        "runtime_contract_major": 1,
        "qualification_revision": "qual-segmented-v1",
    }
    manifest_path = tmp_path / "fast-stage1-segmented.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, manifest


def test_reads_exact_three_segment_package(tmp_path) -> None:
    _, manifest = _write_package(tmp_path)

    package = read_fast_stage1_segmented_package(tmp_path)

    assert [(item.start_index, item.end_index) for item in package.segments] == list(_SPANS)
    assert package.schedule == (1.0, 0.98125, 0.909375, 0.421875, 0.0)
    assert package.transformer_sha256 == manifest["transformer_sha256"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(capability="other"), "unsupported"),
        (lambda value: value.update(base_revision="main"), "immutable"),
        (lambda value: value.update(schedule=[1.0, 0.9, 0.4, 0.0]), "schedule"),
        (lambda value: value.update(schedule=[True, 0.98125, 0.909375, 0.421875, 0.0]), "schedule"),
        (lambda value: value.update(schema_version=True), "schema_version"),
        (lambda value: value.update(runtime_contract_major=True), "runtime contract"),
        (lambda value: value["segments"].reverse(), "segment span"),
    ],
)
def test_rejects_manifest_contract_changes(tmp_path, mutation, match) -> None:
    path, manifest = _write_package(tmp_path)
    mutation(manifest)
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        read_fast_stage1_segmented_package(tmp_path)


def test_rejects_adapter_and_base_digest_mismatches(tmp_path) -> None:
    path, manifest = _write_package(tmp_path)
    manifest["segments"][0]["adapter_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="adapter digest mismatch"):
        read_fast_stage1_segmented_package(tmp_path)

    _, manifest = _write_package(tmp_path)
    manifest["transformer_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="transformer content digest mismatch"):
        read_fast_stage1_segmented_package(tmp_path)


def test_rejects_segment_with_wrong_curriculum_metadata(tmp_path) -> None:
    path, manifest = _write_package(tmp_path)
    segment = tmp_path / manifest["segments"][1]["adapter_file"]
    save_file(
        {
            "block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "block.to_q.lora_B.weight": np.zeros((8, 2), dtype=np.float32),
        },
        segment,
        metadata={"distillation": "stage1_transition"},
    )
    manifest["segments"][1]["adapter_sha256"] = hashlib.sha256(segment.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="incompatible stage1_sigma"):
        read_fast_stage1_segmented_package(tmp_path)


def test_adapter_symlinks_require_diagnostic_qualification(tmp_path) -> None:
    path, manifest = _write_package(tmp_path)
    source = tmp_path / manifest["segments"][0]["adapter_file"]
    external = tmp_path.parent / f"{tmp_path.name}-external.safetensors"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)

    with pytest.raises(ValueError, match="symlinks require a diagnostic"):
        read_fast_stage1_segmented_package(tmp_path)

    manifest["qualification_revision"] = "diagnostic-smoke"
    path.write_text(json.dumps(manifest))
    assert read_fast_stage1_segmented_package(tmp_path).segments[0].adapter_path == source
