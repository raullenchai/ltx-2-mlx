from scripts.run_stage1_compressed_qualification import pilot_gate


def _result(video_change=-20.0, audio_change=-30.0, sample_ratio=0.8):
    return {
        "video_mse_change_percent": video_change,
        "audio_mse_change_percent": audio_change,
        "student": {
            "samples": [
                {"video": {"mse": sample_ratio}, "audio": {"mse": sample_ratio}},
                {"video": {"mse": sample_ratio}, "audio": {"mse": sample_ratio}},
            ]
        },
        "baseline": {
            "samples": [
                {"video": {"mse": 1.0}, "audio": {"mse": 1.0}},
                {"video": {"mse": 1.0}, "audio": {"mse": 1.0}},
            ]
        },
    }


def test_pilot_gate_accepts_material_mean_gain_without_sample_regression() -> None:
    assert pilot_gate(_result(), minimum_improvement_percent=10.0, max_sample_regression_percent=5.0) == []


def test_pilot_gate_rejects_weak_mean_or_single_sample_regression() -> None:
    weak = pilot_gate(_result(video_change=-3.0), minimum_improvement_percent=10.0, max_sample_regression_percent=5.0)
    assert any("video mean" in reason for reason in weak)

    regression = pilot_gate(
        _result(sample_ratio=1.06), minimum_improvement_percent=10.0, max_sample_regression_percent=5.0
    )
    assert any("sample 0 video" in reason for reason in regression)
