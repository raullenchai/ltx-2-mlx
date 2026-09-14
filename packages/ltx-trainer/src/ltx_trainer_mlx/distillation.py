"""Utilities shared by few-step LTX distillation strategies."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def euler_step(
    sample: mx.array,
    velocity: mx.array,
    sigma: mx.array | float,
    sigma_next: mx.array | float,
) -> mx.array:
    """Advance a rectified-flow sample between two sigma values."""
    return sample + (mx.array(sigma_next) - mx.array(sigma)) * velocity


def terminal_velocity_target(
    sample: mx.array,
    terminal_sample: mx.array,
    sigma: mx.array | float,
) -> mx.array:
    """Return the one-step velocity that maps ``sample`` to sigma zero.

    LTX uses ``x_sigma = (1 - sigma) * x0 + sigma * noise`` and predicts
    ``velocity = noise - x0``. Euler integration from ``sigma`` to zero is
    therefore ``x_terminal = sample - sigma * velocity``.
    """
    return transition_velocity_target(sample, terminal_sample, sigma, 0.0)


def transition_velocity_target(
    sample: mx.array,
    target_sample: mx.array,
    sigma: mx.array | float,
    target_sigma: mx.array | float,
) -> mx.array:
    """Return the velocity mapping ``sample`` between two shared sigmas."""
    sigma_array = mx.array(sigma)
    target_sigma_array = mx.array(target_sigma)
    if sigma_array.size != 1 or target_sigma_array.size != 1:
        raise ValueError("distillation requires one shared sigma transition")
    sigma_value = float(sigma_array.item())
    target_sigma_value = float(target_sigma_array.item())
    if sigma_value <= 0:
        raise ValueError("sigma must be greater than zero")
    if not 0 <= target_sigma_value < sigma_value:
        raise ValueError("target_sigma must be in [0, sigma)")
    return (sample - target_sample) / (sigma_array - target_sigma_array)


def _lora_modules(model: nn.Module) -> list[Any]:
    """Find LoRA-like modules without depending on a particular tuner class."""
    return [
        module
        for _, module in model.named_modules()
        if hasattr(module, "lora_a") and hasattr(module, "lora_b") and hasattr(module, "scale")
    ]


@contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily disable every LoRA adapter while running the base teacher.

    The original scales are restored even when teacher generation raises. A
    single quantized model can consequently serve as both frozen teacher and
    trainable LoRA student without doubling resident model memory.
    """
    modules = _lora_modules(model)
    original_scales = [module.scale for module in modules]
    try:
        for module in modules:
            module.scale = 0.0
        yield
    finally:
        for module, scale in zip(modules, original_scales, strict=True):
            module.scale = scale
