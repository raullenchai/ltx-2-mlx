import mlx.core as mx

from scripts.analyze_stage1_schedule import chord_error


def test_chord_error_is_zero_for_sigma_linear_path() -> None:
    sigmas = [1.0, 0.75, 0.5, 0.0]
    path = [mx.full((2, 2), sigma) for sigma in sigmas]

    assert chord_error(path, sigmas, (0, 3)) == 0.0


def test_chord_error_detects_curved_intermediate() -> None:
    sigmas = [1.0, 0.5, 0.0]
    path = [mx.ones((2, 2)), mx.ones((2, 2)), mx.zeros((2, 2))]

    assert chord_error(path, sigmas, (0, 2)) > 0.0
