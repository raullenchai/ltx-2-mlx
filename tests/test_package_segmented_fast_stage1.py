from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.package_segmented_fast_stage1 import main

_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
_SPANS = ((0, 3), (3, 5), (5, 7))


def _checkpoint(path, start, end):
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


def _argv(tmp_path, *, qualification="qual-v1", symlink=False):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "transformer-distilled.safetensors").write_bytes(b"base")
    (model / "embedded_config.json").write_text("{}")
    checkpoints = []
    for start, end in _SPANS:
        path = tmp_path / f"source-{start}-{end}.safetensors"
        _checkpoint(path, start, end)
        checkpoints.extend(["--checkpoint", str(path)])
    argv = [
        "package_segmented_fast_stage1.py",
        "--model-dir",
        str(model),
        *checkpoints,
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
        argv.append("--symlink-adapters")
    return argv


def test_packages_three_bound_adapters(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))

    assert main() == 0

    manifest = json.loads((tmp_path / "output" / "fast-stage1-segmented.json").read_text())
    assert manifest["learned_spans"] == [[0, 3], [3, 5], [5, 7]]
    assert manifest["clean_final_span"] == [7, 8]
    assert len(manifest["segments"]) == 3
    assert all(not (tmp_path / "output" / item["adapter_file"]).is_symlink() for item in manifest["segments"])


def test_symlinks_are_diagnostic_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, symlink=True))
    with pytest.raises(ValueError, match="diagnostic"):
        main()

    monkeypatch.setattr(sys, "argv", _argv(tmp_path, qualification="diagnostic-smoke", symlink=True))
    assert main() == 0
    assert all(path.is_symlink() for path in (tmp_path / "output").glob("*.safetensors"))


def test_rejects_wrong_checkpoint_order_before_writing(monkeypatch, tmp_path) -> None:
    argv = _argv(tmp_path)
    first = argv.index("--checkpoint")
    second = argv.index("--checkpoint", first + 1)
    argv[first + 1], argv[second + 1] = argv[second + 1], argv[first + 1]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="incompatible stage1_sigma"):
        main()
    assert not (tmp_path / "output").exists()
