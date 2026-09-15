import mlx.core as mx
import pytest

from ltx_trainer_mlx.distribution_matching import (
    distribution_matching_surrogate_loss,
    fake_score_flow_loss,
    normalized_distribution_matching_gradient,
    rectified_flow_noising,
    rectified_flow_x0,
)


def test_rectified_flow_noising_and_x0_round_trip() -> None:
    clean = mx.array([[[1.0, -2.0], [3.0, 4.0]]])
    noise = mx.array([[[5.0, 2.0], [-1.0, 8.0]]])
    sigma = 0.25
    velocity = noise - clean

    sample = rectified_flow_noising(clean, noise, sigma)

    assert mx.allclose(rectified_flow_x0(sample, velocity, sigma), clean)


def test_distribution_matching_gradient_matches_dmd2_residual_normalization() -> None:
    generated = mx.array([[[1.0, 2.0]]])
    real_x0 = mx.array([[[0.0, 0.0]]])
    fake_x0 = mx.array([[[0.5, 1.0]]])

    gradient = normalized_distribution_matching_gradient(generated, real_x0, fake_x0)

    assert mx.allclose(gradient, mx.array([[[1.0 / 3.0, 2.0 / 3.0]]]))


def test_surrogate_backpropagates_only_the_supplied_field() -> None:
    field = mx.array([[[0.25, -0.5]]])
    gradient = mx.grad(lambda generated: distribution_matching_surrogate_loss(generated, field))(
        mx.array([[[2.0, 3.0]]])
    )

    assert mx.allclose(gradient, field / field.size)


def test_zero_real_residual_is_finite() -> None:
    sample = mx.zeros((1, 2, 3))
    gradient = normalized_distribution_matching_gradient(sample, sample, mx.ones_like(sample))

    assert bool(mx.all(mx.isfinite(gradient)).item())


def test_fake_score_flow_loss_uses_noise_minus_clean_velocity() -> None:
    clean = mx.array([[[1.0, 3.0]]])
    noise = mx.array([[[4.0, -1.0]]])

    assert float(fake_score_flow_loss(noise - clean, clean, noise).item()) == 0.0


@pytest.mark.parametrize("function", [rectified_flow_noising, rectified_flow_x0, fake_score_flow_loss])
def test_flow_primitives_reject_mismatched_shapes(function) -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        function(mx.zeros((1, 2, 3)), mx.zeros((1, 3, 3)), 0.5)
