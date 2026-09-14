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


def test_percent_change_is_json_safe_for_zero_baseline() -> None:
    assert percent_change(1.0, 0.0) is None
    assert percent_change(0.75, 1.0) == -25.0
