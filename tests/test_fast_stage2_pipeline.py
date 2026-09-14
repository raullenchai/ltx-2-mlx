from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import mlx.core as mx
import pytest

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx import distilled
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput


def _state(value: float) -> LatentState:
    latent = mx.full((1, 2, 3), value)
    return LatentState(latent=latent, clean_latent=mx.zeros_like(latent), denoise_mask=mx.ones((1, 2, 1)))


def _pipeline(monkeypatch):
    pipe = object.__new__(DistilledPipeline)
    pipe.dit = "stage1"
    pipe._loaded = True
    pipe._tile_count = None
    pipe._fast_stage2_package = SimpleNamespace(
        adapter_path=Path("/model/fast.safetensors"),
        transformer_path=Path("/model/transformer-distilled.safetensors"),
        contract=SimpleNamespace(lora_alpha=4.0, lora_rank=8, schedule=(0.909375, 0.421875, 0.0)),
    )
    loads = []

    def load(_self, path):
        pending = getattr(_self, "_pending_loras", None)
        loads.append((path, pending))
        return "student" if pending else "base"

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._stage2_model = MethodType(lambda _self, dit, _shape: dit, pipe)
    monkeypatch.setattr(distilled, "aggressive_cleanup", lambda: None)
    monkeypatch.setattr(distilled, "_materialize", lambda *_args: None)
    return pipe, loads


def test_fast_stage2_uses_student_then_clean_base(monkeypatch) -> None:
    pipe, loads = _pipeline(monkeypatch)
    calls = []

    def denoise(**kwargs):
        calls.append((kwargs["model"], kwargs["sigmas"], kwargs["video_state"].latent))
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        )

    monkeypatch.setattr(distilled, "denoise_loop", denoise)
    result = pipe._run_fast_stage2(
        _state(0),
        _state(10),
        mx.zeros((1, 1, 1)),
        mx.zeros((1, 1, 1)),
        latent_shape=(1, 1, 2),
    )

    assert calls[0][:2] == ("student", [0.909375, 0.421875])
    assert calls[1][:2] == ("base", [0.421875, 0.0])
    assert mx.array_equal(calls[1][2], calls[0][2] + 1).item()
    assert mx.array_equal(result.video_latent, calls[0][2] + 2).item()
    assert loads[0][1] == [("/model/fast.safetensors", 0.5)]
    assert loads[1][1] is None
    assert pipe.dit == "base"
    assert not hasattr(pipe, "_pending_loras")


def test_fast_stage2_failure_releases_partial_model(monkeypatch) -> None:
    pipe, _loads = _pipeline(monkeypatch)
    monkeypatch.setattr(distilled, "denoise_loop", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")))

    with pytest.raises(RuntimeError, match="failed"):
        pipe._run_fast_stage2(
            _state(0),
            _state(0),
            mx.zeros((1, 1, 1)),
            mx.zeros((1, 1, 1)),
            latent_shape=(1, 1, 2),
        )

    assert pipe.dit is None
    assert pipe._loaded is False
    assert not hasattr(pipe, "_pending_loras")
