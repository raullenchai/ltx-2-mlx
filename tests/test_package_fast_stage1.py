from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from scripts.package_fast_stage1 import _validate_source_checkpoint, main


def test_rejects_nonfinal_stage1_source_checkpoint() -> None:
    with pytest.raises(ValueError, match="final compressed Stage-1"):
        _validate_source_checkpoint(
            {
                "distillation": "stage1_transition",
                "stage1_sigma": "1.0",
                "stage1_target_sigma": "0.98125",
            }
        )


def test_packages_selected_stage1_schedule(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    transformer = model / "transformer-distilled.safetensors"
    transformer.write_bytes(b"base")
    config = model / "embedded_config.json"
    config.write_text("{}")
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file(
        {
            "block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "block.to_q.lora_B.weight": np.zeros((8, 2), dtype=np.float32),
        },
        checkpoint,
        metadata={
            "distillation": "stage1_transition",
            "stage1_sigma": "0.421875",
            "stage1_target_sigma": "0.0",
            "stage1_video_start_latents_dir": "stage1_video_step_07",
            "stage1_video_target_latents_dir": "stage1_video_step_08",
            "stage1_audio_start_latents_dir": "stage1_audio_step_07",
            "stage1_audio_target_latents_dir": "stage1_audio_step_08",
            "lora_rank": "2",
            "lora_alpha": "2",
        },
    )
    output = tmp_path / "package"
    revision = "a" * 40
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_fast_stage1.py",
            "--model-dir",
            str(model),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--base-model-id",
            "example/model",
            "--base-revision",
            revision,
            "--qualification-revision",
            "qual-v1",
        ],
    )

    assert main() == 0
    adapter = output / "fast-stage1.safetensors"
    with safe_open(adapter, framework="numpy") as source:
        metadata = source.metadata()
    assert metadata["fast_stage1_capability"] == "ltx_stage1_compressed_v1"
    assert json.loads(metadata["fast_stage1_schedule"]) == [1.0, 0.98125, 0.909375, 0.421875, 0.0]
    assert json.loads(metadata["fast_stage1_noise_step_indices"]) == [0, 3, 5, 7]
    assert metadata["transformer_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert metadata["transformer_sha256"] == hashlib.sha256(transformer.read_bytes()).hexdigest()

    manifest = json.loads((output / "fast-stage1.json").read_text())
    assert manifest["adapter_sha256"] == hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert manifest["transformer_sha256"] == metadata["transformer_sha256"]
