from pathlib import Path

from scripts.train_stage1_compressed_curriculum import PHASES, PRIMARY_PHASES, REPLAY_PHASES, build_config


def test_curriculum_uses_selected_boundaries_and_original_noise_lanes() -> None:
    assert [(p.start_index, p.target_index) for p in PRIMARY_PHASES] == [(0, 3), (3, 5), (5, 7), (7, 8)]
    assert [p.noise_step_index for p in PRIMARY_PHASES] == [0, 3, 5, None]
    assert [(p.sigma, p.target_sigma) for p in PRIMARY_PHASES] == [
        (1.0, 0.98125),
        (0.98125, 0.909375),
        (0.909375, 0.421875),
        (0.421875, 0.0),
    ]
    assert len(PHASES) == 8


def test_ancestral_phase_config_is_noise_coupled_and_resumes() -> None:
    checkpoint = Path("/tmp/previous.safetensors")
    phase = PRIMARY_PHASES[1]
    config = build_config(
        phase=phase,
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/stage1"),
        output=Path("/output/phase"),
        load_checkpoint=checkpoint,
    )

    assert config["model"]["load_checkpoint"] == str(checkpoint)
    assert config["training_strategy"] == {
        "name": "stage1_transition_distill",
        "sigma": 0.98125,
        "target_sigma": 0.909375,
        "video_start_latents_dir": "stage1_video_step_03",
        "video_terminal_latents_dir": "stage1_video_step_05",
        "audio_start_latents_dir": "stage1_audio_step_03",
        "audio_terminal_latents_dir": "stage1_audio_step_05",
        "conditions_dir": "stage1_conditions",
        "video_loss_weight": 1.0,
        "audio_loss_weight": 1.0,
        "ancestral_noise_step_index": 3,
        "ancestral_noise_total_steps": 8,
        "ancestral_eta": 1.0,
        "ancestral_s_noise": 1.0,
    }


def test_terminal_phase_draws_no_noise_and_replay_is_low_lr() -> None:
    config = build_config(
        phase=PRIMARY_PHASES[-1],
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/stage1"),
        output=Path("/output/terminal"),
        load_checkpoint=None,
    )
    assert config["training_strategy"]["target_sigma"] == 0.0
    assert "ancestral_noise_step_index" not in config["training_strategy"]
    assert all(phase.learning_rate == 1.0e-6 and phase.steps == 24 for phase in REPLAY_PHASES)
