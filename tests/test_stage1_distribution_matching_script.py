import random
from pathlib import Path

import mlx.core as mx
import pytest

from scripts.train_stage1_distribution_matching import (
    DEFAULT_SCORE_SIGMAS,
    build_dmd_config,
    sample_decode_crop,
    sample_score_sigma,
)


def test_build_dmd_config_is_a_3_to_7_span_v2_pilot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    config = build_dmd_config(
        model=model,
        transformer_file="transformer.safetensors",
        data=data,
        output=tmp_path / "output",
        checkpoint=None,
        steps=7,
        rank=4,
        learning_rate=1e-6,
        target_modules=None,
    )

    assert config["training_strategy"]["sigma"] == 0.98125
    assert config["training_strategy"]["target_sigma"] == 0.421875
    assert config["training_strategy"]["curriculum_noise_coupling"] == "span-v2"
    assert config["lora"]["rank"] == 4
    assert config["optimization"]["steps"] == 7


@pytest.mark.parametrize("steps,rank,learning_rate", [(0, 4, 1e-6), (1, 0, 1e-6), (1, 4, 0)])
def test_build_dmd_config_rejects_non_positive_values(
    tmp_path: Path, steps: int, rank: int, learning_rate: float
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_dmd_config(
            model=tmp_path,
            transformer_file="transformer.safetensors",
            data=tmp_path,
            output=tmp_path,
            checkpoint=None,
            steps=steps,
            rank=rank,
            learning_rate=learning_rate,
            target_modules=None,
        )


def test_score_sigma_sampler_stays_on_native_non_terminal_schedule() -> None:
    rng = random.Random(7)
    observed = {sample_score_sigma(rng) for _ in range(100)}

    assert observed <= set(DEFAULT_SCORE_SIGMAS)
    assert 0.0 not in observed
    assert 1.0 not in observed


def test_score_sigma_sampler_rejects_terminal_choices() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        sample_score_sigma(random.Random(1), (0.0, 0.5))


def test_decode_crop_sampler_is_seeded_and_stays_inside_grid() -> None:
    batch = {
        "video_start": {
            "num_frames": mx.array([5]),
            "height": mx.array([6]),
            "width": mx.array([7]),
        }
    }

    crop = sample_decode_crop(random.Random(3), batch, (3, 4, 4))

    assert crop == sample_decode_crop(random.Random(3), batch, (3, 4, 4))
    assert all(start >= 0 for start in crop[:3])
    assert all(start + size <= total for start, size, total in zip(crop[:3], crop[3:], (5, 6, 7)))


def test_decode_crop_sampler_rejects_oversize_window() -> None:
    batch = {
        "video_start": {
            "num_frames": mx.array([2]),
            "height": mx.array([2]),
            "width": mx.array([2]),
        }
    }
    with pytest.raises(ValueError, match="fit inside"):
        sample_decode_crop(random.Random(1), batch, (3, 1, 1))
