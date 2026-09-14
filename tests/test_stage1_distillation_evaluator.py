from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_trainer_mlx.stage1_distillation_evaluator import read_stage1_checkpoint_metadata


def _checkpoint(tmp_path, **overrides):
    metadata = {
        "distillation": "stage1_transition",
        "stage1_sigma": "0.725",
        "stage1_target_sigma": "0.0",
        "stage1_video_start_latents_dir": "stage1_video_step_06",
        "stage1_video_target_latents_dir": "stage1_video_step_08",
        "stage1_audio_start_latents_dir": "stage1_audio_step_06",
        "stage1_audio_target_latents_dir": "stage1_audio_step_08",
        "stage1_conditions_dir": "stage1_conditions",
    }
    metadata.update(overrides)
    path = tmp_path / "checkpoint.safetensors"
    save_file({"weight": np.zeros((1,), dtype=np.float32)}, path, metadata=metadata)
    return path


def test_reads_stage1_transition_metadata(tmp_path) -> None:
    metadata = read_stage1_checkpoint_metadata(_checkpoint(tmp_path))

    assert metadata["stage1_sigma"] == "0.725"
    assert metadata["stage1_target_sigma"] == "0.0"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"distillation": "other"}, "not marked as stage1_transition"),
        ({"stage1_sigma": "0"}, "target sigma < start sigma"),
        ({"stage1_target_sigma": "0.8"}, "target sigma < start sigma"),
    ],
)
def test_rejects_invalid_stage1_transition_metadata(tmp_path, overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        read_stage1_checkpoint_metadata(_checkpoint(tmp_path, **overrides))
