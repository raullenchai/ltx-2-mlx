from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_core_mlx.loader.fast_stage1 import read_fast_stage1_contract, read_fast_stage1_package

_REVISION = "a" * 40
_CONFIG_SHA256 = "b" * 64


def _metadata(**overrides: str) -> dict[str, str]:
    values = {
        "fast_stage1_capability": "ltx_stage1_compressed_v1",
        "fast_stage1_schedule": "[1.0,0.98125,0.909375,0.421875,0.0]",
        "fast_stage1_noise_step_indices": "[0,3,5,7]",
        "fast_stage1_noise_total_steps": "8",
        "stage1_steps": "4",
        "stage1_sampler": "ancestral_compressed",
        "stage1_ancestral_eta": "1.0",
        "stage1_ancestral_s_noise": "1.0",
        "lora_rank": "2",
        "lora_alpha": "2",
        "base_model_id": "example/ltx-2.5-mlx-q8",
        "base_revision": _REVISION,
        "transformer_file": "transformer-distilled.safetensors",
        "transformer_config_sha256": _CONFIG_SHA256,
        "pipeline_family": "distilled_two_stage_ltx25",
        "runtime_contract_major": "1",
        "qualification_revision": "qual-v1",
    }
    values.update(overrides)
    return values


def _checkpoint(tmp_path, metadata=None):
    path = tmp_path / "fast-stage1.safetensors"
    save_file(
        {
            "block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "block.to_q.lora_B.weight": np.zeros((8, 2), dtype=np.float32),
        },
        path,
        metadata=metadata or _metadata(),
    )
    return path


def _read(path, **overrides):
    kwargs = {
        "base_model_id": "example/ltx-2.5-mlx-q8",
        "base_revision": _REVISION,
        "transformer_file": "transformer-distilled.safetensors",
        "transformer_config_sha256": _CONFIG_SHA256,
    }
    kwargs.update(overrides)
    return read_fast_stage1_contract(path, **kwargs)


def test_reads_portable_compressed_stage1_contract(tmp_path) -> None:
    path = _checkpoint(tmp_path)
    contract = _read(path, expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    assert contract.schedule == (1.0, 0.98125, 0.909375, 0.421875, 0.0)
    assert contract.noise_step_indices == (0, 3, 5, 7)
    assert contract.noise_total_steps == 8
    assert contract.lora_rank == 2


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fast_stage1_capability": "other"}, "unsupported"),
        ({"fast_stage1_schedule": "[1,0]"}, "five-sigma"),
        ({"fast_stage1_schedule": "[0.9,0.8,0.7,0.6,0]"}, "one to zero"),
        ({"fast_stage1_schedule": "[1,0.8,0.8,0.6,0]"}, "decrease strictly"),
        ({"stage1_steps": "3"}, "stage1_steps=4"),
        ({"fast_stage1_noise_step_indices": "[0,3,5]"}, "one lane per transition"),
        ({"fast_stage1_noise_step_indices": "[1,3,5,7]"}, "start at zero"),
        ({"fast_stage1_noise_step_indices": "[0,3,3,7]"}, "increase strictly"),
        ({"fast_stage1_noise_step_indices": "[0,3,5,8]"}, "exceed"),
        ({"stage1_sampler": "euler"}, "ancestral_compressed"),
        ({"stage1_ancestral_eta": "0.5"}, "eta=1"),
        ({"stage1_ancestral_s_noise": "0.5"}, "s_noise=1"),
        ({"base_revision": "main"}, "immutable hexadecimal revision"),
        ({"runtime_contract_major": "2"}, "runtime contract is incompatible"),
    ],
)
def test_rejects_malformed_stage1_contract(tmp_path, overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _read(_checkpoint(tmp_path, _metadata(**overrides)))


def test_reads_self_contained_stage1_package(tmp_path) -> None:
    config = tmp_path / "embedded_config.json"
    config.write_text('{"transformer":{"model_version":"2.5.0"}}')
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    adapter = _checkpoint(tmp_path, _metadata(transformer_config_sha256=config_sha256))
    transformer = tmp_path / "transformer-distilled.safetensors"
    transformer.write_bytes(b"base placeholder")
    manifest = {
        "schema_version": 1,
        "adapter_file": adapter.name,
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "base_model_id": "example/ltx-2.5-mlx-q8",
        "base_revision": _REVISION,
        "transformer_file": transformer.name,
    }
    (tmp_path / "fast-stage1.json").write_text(json.dumps(manifest))

    package = read_fast_stage1_package(tmp_path)
    assert package.adapter_path == adapter
    assert package.transformer_path == transformer


def test_rejects_stage1_artifact_digest_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        _read(_checkpoint(tmp_path), expected_artifact_sha256="0" * 64)
