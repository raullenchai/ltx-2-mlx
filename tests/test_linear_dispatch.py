"""Tests for the opt-in quantized-linear inference dispatch."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.linear import linear


def _quantized_linear() -> nn.QuantizedLinear:
    layer = nn.Linear(64, 32)
    nn.quantize(layer, group_size=32, bits=8)
    return layer


def test_dequant_matmul_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LTX2_DEQUANT_MATMUL_MIN_TOKENS", raising=False)
    layer = _quantized_linear()
    x = mx.random.normal((1, 8, 64))

    expected = layer(x)
    actual = linear(layer, x)

    assert mx.array_equal(actual, expected).item()


def test_dequant_matmul_matches_quantized_linear(monkeypatch) -> None:
    monkeypatch.setenv("LTX2_DEQUANT_MATMUL_MIN_TOKENS", "8")
    layer = _quantized_linear()
    x = mx.random.normal((1, 8, 64))

    expected = layer(x)
    actual = linear(layer, x)

    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


def test_dequant_matmul_keeps_small_inputs_on_native_path(monkeypatch) -> None:
    monkeypatch.setenv("LTX2_DEQUANT_MATMUL_MIN_TOKENS", "9")
    layer = _quantized_linear()
    x = mx.random.normal((1, 8, 64))

    expected = layer(x)
    actual = linear(layer, x)

    assert mx.array_equal(actual, expected).item()


def test_dequant_matmul_keeps_unquantized_layers_on_native_path(monkeypatch) -> None:
    monkeypatch.setenv("LTX2_DEQUANT_MATMUL_MIN_TOKENS", "1")
    layer = nn.Linear(64, 32)
    x = mx.random.normal((1, 8, 64))

    assert mx.array_equal(linear(layer, x), layer(x)).item()


def test_invalid_threshold_disables_override(monkeypatch) -> None:
    monkeypatch.setenv("LTX2_DEQUANT_MATMUL_MIN_TOKENS", "not-an-integer")
    layer = _quantized_linear()
    x = mx.random.normal((1, 8, 64))

    assert mx.array_equal(linear(layer, x), layer(x)).item()
