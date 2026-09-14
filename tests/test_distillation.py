from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_trainer_mlx.distillation import euler_step, lora_disabled, terminal_velocity_target


class _Adapter(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.lora_a = mx.zeros((1, 1))
        self.lora_b = mx.zeros((1, 1))
        self.scale = scale


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = _Adapter(2.0)
        self.nested = [_Adapter(3.0)]
        self.plain = nn.Linear(1, 1)


def test_terminal_velocity_target_reaches_teacher_terminal() -> None:
    sample = mx.array([[[2.0, -1.0], [0.5, 4.0]]])
    terminal = mx.array([[[0.25, 0.5], [-2.0, 1.0]]])
    sigma = mx.array(0.909375)

    velocity = terminal_velocity_target(sample, terminal, sigma)
    reconstructed = euler_step(sample, velocity, sigma, 0.0)

    assert mx.allclose(reconstructed, terminal, atol=1e-6).item()


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_terminal_velocity_target_rejects_non_positive_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        terminal_velocity_target(mx.zeros((1,)), mx.zeros((1,)), sigma)


def test_terminal_velocity_target_rejects_per_sample_sigma() -> None:
    with pytest.raises(ValueError, match="one shared sigma"):
        terminal_velocity_target(mx.zeros((2, 1)), mx.zeros((2, 1)), mx.array([0.9, 0.8]))


def test_lora_disabled_restores_scales_after_exception() -> None:
    model = _Model()

    with pytest.raises(RuntimeError, match="teacher failed"), lora_disabled(model):
        assert model.first.scale == 0.0
        assert model.nested[0].scale == 0.0
        raise RuntimeError("teacher failed")

    assert model.first.scale == 2.0
    assert model.nested[0].scale == 3.0
