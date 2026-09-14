from pathlib import Path

from scripts.train_stage2_terminal_curriculum import build_config


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
