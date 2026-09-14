from __future__ import annotations

import pytest

from scripts.compare_stage2_distillation import compare_results


def _result(video=(0.5, 1.5), audio=(0.8, 1.0)):
    samples = []
    for index, (video_mse, audio_mse) in enumerate(zip(video, audio, strict=True)):
        samples.append({"index": index, "video": {"mse": video_mse}, "audio": {"mse": audio_mse}})
    return {
        "schedule": [0.909375, 0.0],
        "aggregate": {
            "video": {"mse": sum(video) / len(video)},
            "audio": {"mse": sum(audio) / len(audio)},
        },
        "per_sample": samples,
    }


def test_compares_means_and_worst_paired_sample() -> None:
    result = compare_results(_result(), _result(video=(1.0, 1.0), audio=(1.0, 1.0)))

    assert result["modalities"]["video"]["mean_mse_change_percent"] == 0.0
    assert result["modalities"]["video"]["regressed_sample_count"] == 1
    assert result["modalities"]["video"]["worst_sample_index"] == 1
    assert result["modalities"]["video"]["worst_sample_mse_change_percent"] == 50.0
    assert result["modalities"]["audio"]["mean_mse_change_percent"] == pytest.approx(-10.0)


def test_matches_samples_by_index_not_input_order() -> None:
    baseline = _result(video=(1.0, 2.0))
    baseline["per_sample"].reverse()

    result = compare_results(_result(video=(0.5, 1.0)), baseline)

    assert [sample["index"] for sample in result["per_sample"]] == [0, 1]
    assert result["modalities"]["video"]["mean_mse_change_percent"] == -50.0


def test_rejects_unpaired_or_duplicate_results() -> None:
    mismatch = _result()
    mismatch["per_sample"][1]["index"] = 4
    with pytest.raises(ValueError, match="indices do not match"):
        compare_results(_result(), mismatch)

    duplicate = _result()
    duplicate["per_sample"][1]["index"] = 0
    with pytest.raises(ValueError, match="duplicate sample index"):
        compare_results(duplicate, _result())


def test_rejects_different_schedules() -> None:
    baseline = _result()
    baseline["schedule"] = [0.909375, 0.421875, 0.0]
    with pytest.raises(ValueError, match="schedules do not match"):
        compare_results(_result(), baseline)
