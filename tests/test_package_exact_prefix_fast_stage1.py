from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_core_mlx.loader.fast_stage1_segmented import read_fast_stage1_segmented_package
from scripts.package_exact_prefix_fast_stage1 import main

_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)


def _argv(tmp_path, *, mode="independent", qualification="qual-v1", symlink=False):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "transformer-distilled.safetensors").write_bytes(b"base")
    (model / "embedded_config.json").write_text("{}")
    checkpoint = tmp_path / "source-3-7.safetensors"
    save_file(
        {
            "block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "block.to_q.lora_B.weight": np.zeros((8, 2), dtype=np.float32),
        },
        checkpoint,
        metadata={
            "distillation": "stage1_transition",
            "stage1_sigma": str(_SIGMAS[3]),
            "stage1_target_sigma": str(_SIGMAS[7]),
            "stage1_video_start_latents_dir": "stage1_video_step_03",
            "stage1_video_target_latents_dir": "stage1_video_step_07",
            "stage1_audio_start_latents_dir": "stage1_audio_step_03",
            "stage1_audio_target_latents_dir": "stage1_audio_step_07",
            "stage1_curriculum_noise_coupling": "span-v2",
            "stage1_curriculum_adapter_mode": mode,
            "stage1_sampler": "ancestral_span_v2",
            "stage1_noise_step_index": "3",
            "stage1_noise_step_end_index": "7",
            "stage1_noise_total_steps": "8",
            "stage1_noise_reference_sigmas": json.dumps(_SIGMAS),
            "lora_rank": "2",
            "lora_alpha": "2",
        },
    )
    argv = [
        "package_exact_prefix_fast_stage1.py",
        "--model-dir",
        str(model),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(tmp_path / "output"),
        "--base-model-id",
        "example/model",
        "--base-revision",
        "a" * 40,
        "--qualification-revision",
        qualification,
    ]
    if symlink:
        argv.append("--symlink-adapter")
    return argv


def test_packages_exact_prefix_middle_route(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert main() == 0

    output = tmp_path / "output"
    manifest = json.loads((output / "fast-stage1-exact-prefix.json").read_text())
    assert manifest["capability"] == "ltx_stage1_exact_prefix_middle_span_v1"
    assert manifest["execution_spans"] == [[0, 1], [1, 2], [2, 3], [3, 7]]
    assert manifest["learned_spans"] == [[3, 7]]
    assert manifest["clean_final_span"] == [7, 8]
    shutil.copy2(tmp_path / "model" / "transformer-distilled.safetensors", output)
    shutil.copy2(tmp_path / "model" / "embedded_config.json", output)
    package = read_fast_stage1_segmented_package(output, "fast-stage1-exact-prefix.json")
    assert [segment.adapter_path is None for segment in package.segments] == [True, True, True, False]


def test_rejects_non_independent_middle_checkpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, mode="shared"))

    with pytest.raises(ValueError, match="independently"):
        main()
    assert not (tmp_path / "output").exists()


def test_exact_prefix_symlink_is_diagnostic_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, symlink=True))
    with pytest.raises(ValueError, match="diagnostic"):
        main()

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, qualification="diagnostic-smoke", symlink=True),
    )
    assert main() == 0
    assert (tmp_path / "output" / "fast-stage1-middle-3-7.safetensors").is_symlink()
