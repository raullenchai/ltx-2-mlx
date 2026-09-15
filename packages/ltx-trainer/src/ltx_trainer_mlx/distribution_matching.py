"""Math primitives for experimental rectified-flow distribution matching.

These functions intentionally do not own models or optimizers. They make the
real-score, fake-score, and generator gradient boundaries explicit before the
experimental alternating trainer is connected to the production trainer.
"""

from __future__ import annotations

import mlx.core as mx


def rectified_flow_noising(clean: mx.array, noise: mx.array, sigma: mx.array | float) -> mx.array:
    """Interpolate a clean sample and Gaussian noise at RF time ``sigma``."""
    if clean.shape != noise.shape:
        raise ValueError("clean sample and noise shapes must match")
    sigma_array = mx.array(sigma).reshape(-1, 1, 1)
    if sigma_array.shape[0] not in (1, clean.shape[0]):
        raise ValueError("sigma must be scalar or contain one value per batch item")
    return (1.0 - sigma_array) * clean + sigma_array * noise


def rectified_flow_x0(sample: mx.array, velocity: mx.array, sigma: mx.array | float) -> mx.array:
    """Convert LTX velocity prediction at ``sigma`` to its clean prediction."""
    if sample.shape != velocity.shape:
        raise ValueError("sample and velocity shapes must match")
    sigma_array = mx.array(sigma).reshape(-1, 1, 1)
    if sigma_array.shape[0] not in (1, sample.shape[0]):
        raise ValueError("sigma must be scalar or contain one value per batch item")
    return sample - sigma_array * velocity


def normalized_distribution_matching_gradient(
    generated: mx.array,
    real_x0: mx.array,
    fake_x0: mx.array,
    *,
    eps: float = 1e-6,
) -> mx.array:
    """Return the detached DMD generator gradient from real/fake predictions.

    This follows DMD2's residual normalization: ``(p_real - p_fake) /
    mean(abs(p_real))``, where ``p = generated - predicted_x0``. The returned
    tensor is detached so score-model parameters never receive generator-turn
    gradients.
    """
    if generated.shape != real_x0.shape or generated.shape != fake_x0.shape:
        raise ValueError("generated, real_x0, and fake_x0 shapes must match")
    reduction_axes = tuple(range(1, generated.ndim))
    real_residual = generated - real_x0
    fake_residual = generated - fake_x0
    normalizer = mx.maximum(mx.mean(mx.abs(real_residual), axis=reduction_axes, keepdims=True), eps)
    gradient = (real_residual - fake_residual) / normalizer
    return mx.stop_gradient(mx.nan_to_num(gradient))


def distribution_matching_surrogate_loss(generated: mx.array, gradient: mx.array) -> mx.array:
    """Create a scalar whose autodiff gradient equals the supplied DMD field."""
    if generated.shape != gradient.shape:
        raise ValueError("generated sample and DMD gradient shapes must match")
    target = mx.stop_gradient(generated - gradient)
    return 0.5 * mx.mean(mx.square(generated - target))


def fake_score_flow_loss(predicted_velocity: mx.array, clean: mx.array, noise: mx.array) -> mx.array:
    """Train the fake score on detached generator samples in RF velocity form."""
    if predicted_velocity.shape != clean.shape or predicted_velocity.shape != noise.shape:
        raise ValueError("predicted velocity, clean sample, and noise shapes must match")
    target_velocity = noise - mx.stop_gradient(clean)
    return mx.mean(mx.square(predicted_velocity - target_velocity))
