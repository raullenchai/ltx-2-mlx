from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_core_mlx.loader.fast_stage2 import read_fast_stage2_contract, read_fast_stage2_package

_REVISION = "a" * 40
_CONFIG_SHA256 = "b" * 64


def _metadata(**overrides: str) -> dict[str, str]:
    values = {
        "fast_stage2_capability": "ltx_stage2_transition_v1",
        "fast_stage2_schedule": json.dumps([0.909375, 0.421875, 0.0]),
        "stage2_sigma": "0.909375",
        "stage2_target_sigma": "0.421875",
        "stage2_steps": "2",
        "lora_rank": "2",
        "lora_alpha": "2",
        "base_model_id": "example/ltx-2.5-mlx-q8",
        "base_revision": _REVISION,
        "transformer_file": "transformer-distilled.safetensors",
        "transformer_config_sha256": _CONFIG_SHA256,
        "runtime_contract_major": "1",
        "qualification_revision": "qual-v1",
    }
    values.update(overrides)
    return values


def _checkpoint(tmp_path, metadata: dict[str, str] | None = None, *, orphan: bool = False):
    path = tmp_path / "fast-stage2.safetensors"
    tensors = {"block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32)}
    if not orphan:
        tensors["block.to_q.lora_B.weight"] = np.zeros((8, 2), dtype=np.float32)
    save_file(tensors, path, metadata=metadata or _metadata())
    return path


def _read(path, **overrides):
    kwargs = {
        "base_model_id": "example/ltx-2.5-mlx-q8",
        "base_revision": _REVISION,
        "transformer_file": "transformer-distilled.safetensors",
        "transformer_config_sha256": _CONFIG_SHA256,
    }
    kwargs.update(overrides)
    return read_fast_stage2_contract(path, **kwargs)


def test_reads_portable_fast_stage2_contract(tmp_path) -> None:
    path = _checkpoint(tmp_path)

    contract = _read(path, expected_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    assert contract.capability == "ltx_stage2_transition_v1"
    assert contract.schedule == (0.909375, 0.421875, 0.0)
    assert contract.lora_rank == 2
    assert contract.runtime_contract_major == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fast_stage2_capability": ""}, "missing 'fast_stage2_capability'"),
        ({"fast_stage2_schedule": "[0.9, 0.5]"}, "three-sigma"),
        ({"fast_stage2_schedule": "[0.9, 0.9, 0]"}, "decrease strictly"),
        ({"stage2_sigma": "0.8"}, "stage2_sigma does not match"),
        ({"stage2_target_sigma": "0.4"}, "stage2_target_sigma does not match"),
        ({"stage2_steps": "1"}, "stage2_steps=2"),
        ({"base_revision": "main"}, "immutable hexadecimal revision"),
        ({"runtime_contract_major": "2"}, "runtime contract is incompatible"),
    ],
)
def test_rejects_malformed_contract(tmp_path, overrides, match) -> None:
    path = _checkpoint(tmp_path, _metadata(**overrides))

    with pytest.raises(ValueError, match=match):
        _read(path)


@pytest.mark.parametrize(
    ("override", "value", "match"),
    [
        ("base_model_id", "other/model", "model identifier mismatch"),
        ("base_revision", "c" * 40, "base revision mismatch"),
        ("transformer_file", "other.safetensors", "transformer filename mismatch"),
        ("transformer_config_sha256", "d" * 64, "config fingerprint mismatch"),
    ],
)
def test_rejects_base_checkpoint_mismatch(tmp_path, override, value, match) -> None:
    path = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match=match):
        _read(path, **{override: value})


def test_rejects_artifact_digest_mismatch(tmp_path) -> None:
    path = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        _read(path, expected_artifact_sha256="0" * 64)


def test_rejects_incomplete_or_wrong_rank_lora_pairs(tmp_path) -> None:
    with pytest.raises(ValueError, match="incomplete LoRA tensor pair"):
        _read(_checkpoint(tmp_path, orphan=True))

    with pytest.raises(ValueError, match="does not match declared rank"):
        _read(_checkpoint(tmp_path, _metadata(lora_rank="4")))


def test_reads_self_contained_fast_stage2_package(tmp_path) -> None:
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
    (tmp_path / "fast-stage2.json").write_text(json.dumps(manifest))

    package = read_fast_stage2_package(tmp_path)

    assert package.adapter_path == adapter
    assert package.transformer_path == transformer
    assert package.contract.schedule == (0.909375, 0.421875, 0.0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "schema_version=1"),
        ("adapter_file", "../adapter.safetensors", "one local .safetensors"),
        ("transformer_file", "/tmp/base.safetensors", "one local .safetensors"),
    ],
)
def test_rejects_invalid_fast_stage2_package_paths(tmp_path, field, value, match) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
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
        field: value,
    }
    (tmp_path / "fast-stage2.json").write_text(json.dumps(manifest))

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        read_fast_stage2_package(tmp_path)
