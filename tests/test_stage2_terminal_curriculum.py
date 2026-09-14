from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.train_stage2_terminal_curriculum import build_config, validate_terminal_checkpoint


def _checkpoint(path: Path, **overrides: str) -> Path:
    metadata = {
        "distillation": "stage2_terminal",
        "stage2_sigma": "0.909375",
        "stage2_target_sigma": "0.0",
        "stage2_steps": "1",
        "stage2_video_target_latents_dir": "stage2_video_terminal_latents",
        "stage2_audio_target_latents_dir": "stage2_audio_terminal_latents",
        "lora_rank": "8",
        "lora_alpha": "8",
    }
    metadata.update(overrides)
    save_file({"adapter": np.zeros((1,), dtype=np.float32)}, path, metadata=metadata)
    return path


def test_terminal_curriculum_config_uses_exact_terminal_target() -> None:
    checkpoint = Path("/tmp/previous.safetensors")
    config = build_config(
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/468"),
        output=Path("/output/468"),
        steps=100,
        learning_rate=1.0e-4,
        checkpoint_interval=20,
        load_checkpoint=checkpoint,
    )

    assert config["model"]["load_checkpoint"] == str(checkpoint)
    assert config["training_strategy"] == {
        "name": "stage2_terminal_distill",
        "sigma": 0.909375,
        "target_sigma": 0.0,
        "video_terminal_latents_dir": "stage2_video_terminal_latents",
        "audio_terminal_latents_dir": "stage2_audio_terminal_latents",
        "video_loss_weight": 1.0,
        "audio_loss_weight": 1.0,
    }
    assert config["optimization"]["steps"] == 100
    assert config["data"]["preprocessed_data_root"] == "/data/468"


def test_terminal_curriculum_accepts_compatible_resume_checkpoint(tmp_path) -> None:
    validate_terminal_checkpoint(_checkpoint(tmp_path / "checkpoint.safetensors"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"distillation": "stage2_progressive"},
        {"stage2_sigma": "0.8"},
        {"stage2_target_sigma": "0.421875"},
        {"stage2_steps": "2"},
        {"lora_rank": "0"},
    ],
)
def test_terminal_curriculum_rejects_incompatible_resume_checkpoint(tmp_path, overrides) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.safetensors", **overrides)

    with pytest.raises(ValueError, match="checkpoint"):
        validate_terminal_checkpoint(checkpoint)
