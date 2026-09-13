"""Inference-time linear dispatch helpers for Apple Silicon."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

_DEQUANT_MATMUL_ENV = "LTX2_DEQUANT_MATMUL_MIN_TOKENS"


def _dequant_matmul_min_tokens() -> int:
    """Return the opt-in token threshold, or zero when disabled/invalid."""
    raw = os.environ.get(_DEQUANT_MATMUL_ENV, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def linear(layer: nn.Module, x: mx.array) -> mx.array:
    """Apply a linear layer, optionally dequantizing large QMMs first.

    On some Apple GPU / MLX combinations, a BF16 matmul over an explicitly
    dequantized affine weight is faster than ``quantized_matmul`` once the
    flattened token count is large.  Keep this path opt-in because the
    crossover is hardware and MLX-version dependent.

    Set ``LTX2_DEQUANT_MATMUL_MIN_TOKENS`` to the minimum flattened token
    count to enable it.  Small text/audio projections continue to use MLX's
    native quantized linear path.
    """
    threshold = _dequant_matmul_min_tokens()
    token_count = x.size // x.shape[-1]
    if threshold <= 0 or token_count < threshold or not isinstance(layer, nn.QuantizedLinear):
        return layer(x)

    weight = mx.dequantize(
        layer.weight,
        layer.scales,
        layer.biases,
        group_size=layer.group_size,
        bits=layer.bits,
    )
    output = x @ weight.T
    bias = getattr(layer, "bias", None)
    return output if bias is None else output + bias
