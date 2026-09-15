from scripts.evaluate_stage1_compressed_curriculum import metadata_for_phase, percent_change
from scripts.train_stage1_compressed_curriculum import PRIMARY_PHASES


def test_phase_metadata_selects_each_captured_boundary_and_noise_lane() -> None:
    checkpoint = {"lora_rank": "8", "stage1_sampler": "stale", "stage1_noise_step_index": "7"}

    ancestral = metadata_for_phase(checkpoint, PRIMARY_PHASES[2])
    assert ancestral["lora_rank"] == "8"
    assert ancestral["stage1_video_start_latents_dir"] == "stage1_video_step_05"
    assert ancestral["stage1_video_target_latents_dir"] == "stage1_video_step_07"
    assert ancestral["stage1_noise_step_index"] == "5"
    assert ancestral["stage1_noise_total_steps"] == "8"

    terminal = metadata_for_phase(checkpoint, PRIMARY_PHASES[-1])
    assert terminal["stage1_target_sigma"] == "0.0"
    assert "stage1_sampler" not in terminal
    assert "stage1_noise_step_index" not in terminal


def test_phase_metadata_preserves_span_v2_coupling() -> None:
    checkpoint = {
        "lora_rank": "8",
        "stage1_curriculum_noise_coupling": "span-v2",
        "stage1_sampler": "stale",
        "stage1_noise_step_index": "7",
    }

    metadata = metadata_for_phase(checkpoint, PRIMARY_PHASES[1])

    assert metadata["stage1_sampler"] == "ancestral_span_v2"
    assert metadata["stage1_noise_step_index"] == "3"
    assert metadata["stage1_noise_step_end_index"] == "5"
    assert metadata["stage1_noise_reference_sigmas"] == (
        "[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]"
    )


def test_percent_change_is_json_safe_for_zero_baseline() -> None:
    assert percent_change(1.0, 0.0) is None
    assert percent_change(0.75, 1.0) == -25.0
