"""Tests for the opt-in video VAE decoder precision."""

import mlx.core as mx
import pytest

from ltx_pipelines_mlx.utils.blocks import _configure_video_decoder_dtype


class _FakeDecoder:
    def __init__(self) -> None:
        self.params = {"weight": mx.ones((2, 2), dtype=mx.bfloat16)}

    def parameters(self):
        return self.params

    def update(self, params) -> None:
        self.params = params


def test_fp16_casts_decoder_parameters() -> None:
    decoder = _FakeDecoder()

    _configure_video_decoder_dtype(decoder, "fp16")

    assert decoder.params["weight"].dtype == mx.float16


def test_native_preserves_decoder_parameters() -> None:
    decoder = _FakeDecoder()

    _configure_video_decoder_dtype(decoder, "native")

    assert decoder.params["weight"].dtype == mx.bfloat16


def test_invalid_decoder_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="LTX2_VAE_DECODER_DTYPE"):
        _configure_video_decoder_dtype(_FakeDecoder(), "fp32")
