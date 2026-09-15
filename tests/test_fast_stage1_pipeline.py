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


def test_load_applies_only_validated_stage1_adapter() -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe._loaded = False
    pipe.dit = None
    pipe.upsampler = object()
    pipe._fast_stage1_package = SimpleNamespace(
        adapter_path=Path("/model/fast-stage1.safetensors"),
        transformer_path=Path("/model/transformer-distilled.safetensors"),
        contract=SimpleNamespace(lora_alpha=4.0, lora_rank=8),
    )
    pipe._fast_stage1_segmented_package = None
    observed = []

    def load(_self, path):
        observed.append((path, list(_self._pending_loras)))
        return "stage1-student"

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._load_vae_encoder = lambda: None

    pipe.load()

    assert pipe.dit == "stage1-student"
    assert observed == [
        (
            Path("/model/transformer-distilled.safetensors"),
            [("/model/fast-stage1.safetensors", 0.5)],
        )
    ]
    assert not hasattr(pipe, "_pending_loras")
    assert pipe._loaded is True


def _segmented_package() -> SimpleNamespace:
    return SimpleNamespace(
        transformer_path=Path("/model/transformer-distilled.safetensors"),
        schedule=(1.0, 0.98125, 0.909375, 0.421875, 0.0),
        noise_reference_sigmas=(1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
        noise_total_steps=8,
        segments=tuple(
            SimpleNamespace(
                adapter_path=Path(f"/model/segment-{start}-{end}.safetensors"),
                start_index=start,
                end_index=end,
                sigma=sigma,
                target_sigma=target,
                lora_rank=8,
                lora_alpha=4.0,
            )
            for start, end, sigma, target in (
                (0, 3, 1.0, 0.98125),
                (3, 5, 0.98125, 0.909375),
                (5, 7, 0.909375, 0.421875),
            )
        ),
    )


def _exact_prefix_package() -> SimpleNamespace:
    return SimpleNamespace(
        transformer_path=Path("/model/transformer-distilled.safetensors"),
        schedule=(1.0, 0.99375, 0.9875, 0.98125, 0.421875, 0.0),
        noise_reference_sigmas=(1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
        noise_total_steps=8,
        segments=tuple(
            SimpleNamespace(
                adapter_path=Path("/model/middle-3-7.safetensors") if (start, end) == (3, 7) else None,
                start_index=start,
                end_index=end,
                sigma=sigma,
                target_sigma=target,
                lora_rank=4 if (start, end) == (3, 7) else None,
                lora_alpha=4.0 if (start, end) == (3, 7) else None,
            )
            for start, end, sigma, target in (
                (0, 1, 1.0, 0.99375),
                (1, 2, 0.99375, 0.9875),
                (2, 3, 0.9875, 0.98125),
                (3, 7, 0.98125, 0.421875),
            )
        ),
    )


def test_load_applies_first_segment_only() -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe._loaded = False
    pipe.dit = None
    pipe.upsampler = object()
    pipe._fast_stage1_package = None
    pipe._fast_stage1_segmented_package = _segmented_package()
    observed = []

    def load(_self, path):
        observed.append((path, list(_self._pending_loras)))
        return "segment-0-3"

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._load_vae_encoder = lambda: None

    pipe.load()

    assert pipe.dit == "segment-0-3"
    assert observed == [
        (
            Path("/model/transformer-distilled.safetensors"),
            [("/model/segment-0-3.safetensors", 0.5)],
        )
    ]
    assert not hasattr(pipe, "_pending_loras")


def test_load_uses_clean_base_for_exact_prefix() -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe._loaded = False
    pipe.dit = None
    pipe.upsampler = object()
    pipe._fast_stage1_package = None
    pipe._fast_stage1_segmented_package = _exact_prefix_package()
    observed = []

    def load(_self, path):
        observed.append((path, getattr(_self, "_pending_loras", None)))
        return "base"

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._load_vae_encoder = lambda: None

    pipe.load()

    assert pipe.dit == "base"
    assert observed == [(Path("/model/transformer-distilled.safetensors"), None)]


def test_exact_prefix_route_loads_only_middle_adapter(monkeypatch) -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe.dit = "base"
    pipe._loaded = True
    pipe._tile_count = None
    pipe._fast_stage1_segmented_package = _exact_prefix_package()
    loads = []

    def load(_self, path):
        pending = getattr(_self, "_pending_loras", None)
        label = Path(pending[0][0]).stem if pending else "base"
        loads.append((path, list(pending) if pending else None))
        return label

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._stage2_model = MethodType(lambda _self, dit, _shape: dit, pipe)
    calls = []

    def ancestral(**kwargs):
        calls.append((kwargs["model"], kwargs["noise_step_spans"]))
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        )

    monkeypatch.setattr(distilled, "ancestral_denoise_loop", ancestral)
    monkeypatch.setattr(
        distilled,
        "denoise_loop",
        lambda **kwargs: DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        ),
    )
    monkeypatch.setattr(distilled, "aggressive_cleanup", lambda: None)
    monkeypatch.setattr(distilled, "_materialize", lambda *_args: None)

    pipe._run_segmented_fast_stage1(
        _state(0),
        _state(10),
        mx.zeros((1, 1, 1)),
        mx.zeros((1, 1, 1)),
        latent_shape=(1, 1, 2),
        noise_seed=10042,
    )

    assert calls == [
        ("base", [(0, 1)]),
        ("base", [(1, 2)]),
        ("base", [(2, 3)]),
        ("middle-3-7", [(3, 7)]),
    ]
    assert [item[1] for item in loads] == [
        None,
        None,
        [("/model/middle-3-7.safetensors", 1.0)],
        None,
    ]


def test_diagnostic_ablation_can_use_base_for_first_segment() -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe._loaded = False
    pipe.dit = None
    pipe.upsampler = object()
    pipe._fast_stage1_package = None
    pipe._fast_stage1_segmented_package = _segmented_package()
    pipe._diagnostic_base_stage1_spans = frozenset({(0, 3)})
    observed = []

    def load(_self, path):
        observed.append((path, getattr(_self, "_pending_loras", None)))
        return "base"

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._load_vae_encoder = lambda: None

    pipe.load()

    assert pipe.dit == "base"
    assert observed == [(Path("/model/transformer-distilled.safetensors"), None)]


def test_segmented_stage1_swaps_each_student_then_uses_exact_base(monkeypatch) -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe.dit = "segment-0-3"
    pipe._loaded = True
    pipe._tile_count = None
    pipe._fast_stage1_segmented_package = _segmented_package()
    loads = []

    def load(_self, path):
        pending = getattr(_self, "_pending_loras", None)
        label = Path(pending[0][0]).stem if pending else "base"
        loads.append((path, list(pending) if pending else None))
        return label

    pipe._load_transformer_with_optional_streaming = MethodType(load, pipe)
    pipe._stage2_model = MethodType(lambda _self, dit, _shape: dit, pipe)
    calls = []

    def ancestral(**kwargs):
        calls.append(("ancestral", kwargs))
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        )

    def deterministic(**kwargs):
        calls.append(("base", kwargs))
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        )

    monkeypatch.setattr(distilled, "ancestral_denoise_loop", ancestral)
    monkeypatch.setattr(distilled, "denoise_loop", deterministic)
    monkeypatch.setattr(distilled, "aggressive_cleanup", lambda: None)
    monkeypatch.setattr(distilled, "_materialize", lambda *_args: None)

    result = pipe._run_segmented_fast_stage1(
        _state(0),
        _state(10),
        mx.zeros((1, 1, 1)),
        mx.zeros((1, 1, 1)),
        latent_shape=(1, 1, 2),
        noise_seed=10042,
    )

    assert [call[1]["model"] for call in calls] == [
        "segment-0-3",
        "segment-3-5",
        "segment-5-7",
        "base",
    ]
    assert [call[1]["sigmas"] for call in calls] == [
        [1.0, 0.98125],
        [0.98125, 0.909375],
        [0.909375, 0.421875],
        [0.421875, 0.0],
    ]
    assert [call[1]["noise_step_spans"] for call in calls[:3]] == [[(0, 3)], [(3, 5)], [(5, 7)]]
    assert all(call[1]["noise_seed"] == 10042 for call in calls[:3])
    assert mx.array_equal(calls[1][1]["video_state"].latent, mx.ones((1, 2, 3))).item()
    assert mx.array_equal(calls[3][1]["video_state"].latent, mx.full((1, 2, 3), 3)).item()
    assert mx.array_equal(result.video_latent, mx.full((1, 2, 3), 4)).item()
    assert [item[1] for item in loads] == [
        [("/model/segment-3-5.safetensors", 0.5)],
        [("/model/segment-5-7.safetensors", 0.5)],
        None,
    ]
    assert pipe.dit == "base"


def test_segmented_stage1_failure_releases_model(monkeypatch) -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe.dit = "segment-0-3"
    pipe._loaded = True
    pipe._tile_count = None
    pipe._fast_stage1_segmented_package = _segmented_package()
    pipe._stage2_model = MethodType(lambda _self, dit, _shape: dit, pipe)
    pipe._load_transformer_with_optional_streaming = MethodType(lambda _self, _path: "next", pipe)
    monkeypatch.setattr(
        distilled,
        "ancestral_denoise_loop",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    monkeypatch.setattr(distilled, "aggressive_cleanup", lambda: None)

    with pytest.raises(RuntimeError, match="failed"):
        pipe._run_segmented_fast_stage1(
            _state(0),
            _state(0),
            mx.zeros((1, 1, 1)),
            mx.zeros((1, 1, 1)),
            latent_shape=(1, 1, 2),
            noise_seed=1,
        )

    assert pipe.dit is None
    assert pipe._loaded is False


def test_clean_final_stage1_uses_student_then_exact_base(monkeypatch) -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe.dit = "student"
    pipe._loaded = True
    pipe._tile_count = None
    pipe._fast_stage1_package = SimpleNamespace(
        transformer_path=Path("/model/transformer-distilled.safetensors"),
        contract=SimpleNamespace(
            clean_final_transition=True,
            schedule=(1.0, 0.98125, 0.909375, 0.421875, 0.0),
            noise_step_spans=((0, 3), (3, 5), (5, 7), (7, 8)),
            noise_reference_sigmas=(1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
            noise_total_steps=8,
        ),
    )
    loads = []
    pipe._load_transformer_with_optional_streaming = MethodType(
        lambda _self, path: loads.append(path) or "base",
        pipe,
    )
    pipe._stage2_model = MethodType(lambda _self, dit, _shape: dit, pipe)
    calls = []

    def ancestral(**kwargs):
        calls.append(("ancestral", kwargs))
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        )

    def deterministic(**kwargs):
        calls.append(("base", kwargs))
        return DenoiseOutput(
            video_latent=kwargs["video_state"].latent + 1,
            audio_latent=kwargs["audio_state"].latent + 1,
        )

    monkeypatch.setattr(distilled, "ancestral_denoise_loop", ancestral)
    monkeypatch.setattr(distilled, "denoise_loop", deterministic)
    monkeypatch.setattr(distilled, "aggressive_cleanup", lambda: None)
    monkeypatch.setattr(distilled, "_materialize", lambda *_args: None)

    result = pipe._run_clean_final_fast_stage1(
        _state(0),
        _state(10),
        mx.zeros((1, 1, 1)),
        mx.zeros((1, 1, 1)),
        latent_shape=(1, 1, 2),
        noise_seed=10042,
    )

    learned = calls[0][1]
    correction = calls[1][1]
    assert learned["model"] == "student"
    assert learned["sigmas"] == [1.0, 0.98125, 0.909375, 0.421875]
    assert learned["noise_step_spans"] == [(0, 3), (3, 5), (5, 7)]
    assert learned["noise_seed"] == 10042
    assert correction["model"] == "base"
    assert correction["sigmas"] == [0.421875, 0.0]
    assert mx.array_equal(correction["video_state"].latent, mx.ones((1, 2, 3))).item()
    assert mx.array_equal(result.video_latent, mx.full((1, 2, 3), 2)).item()
    assert loads == [Path("/model/transformer-distilled.safetensors")]
    assert pipe.dit == "base"


def test_clean_final_stage1_failure_releases_model(monkeypatch) -> None:
    pipe = object.__new__(DistilledPipeline)
    pipe.dit = "student"
    pipe._loaded = True
    pipe._tile_count = None
    pipe._fast_stage1_package = SimpleNamespace(
        transformer_path=Path("/model/transformer-distilled.safetensors"),
        contract=SimpleNamespace(
            clean_final_transition=True,
            schedule=(1.0, 0.98125, 0.909375, 0.421875, 0.0),
            noise_step_spans=((0, 3), (3, 5), (5, 7), (7, 8)),
            noise_reference_sigmas=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0),
            noise_total_steps=8,
        ),
    )
    pipe._stage2_model = MethodType(lambda _self, dit, _shape: dit, pipe)
    pipe._load_transformer_with_optional_streaming = MethodType(lambda _self, _path: "base", pipe)
    monkeypatch.setattr(
        distilled,
        "ancestral_denoise_loop",
        lambda **kwargs: DenoiseOutput(
            video_latent=kwargs["video_state"].latent,
            audio_latent=kwargs["audio_state"].latent,
        ),
    )
    monkeypatch.setattr(distilled, "denoise_loop", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
    monkeypatch.setattr(distilled, "aggressive_cleanup", lambda: None)
    monkeypatch.setattr(distilled, "_materialize", lambda *_args: None)

    with pytest.raises(RuntimeError, match="failed"):
        pipe._run_clean_final_fast_stage1(
            _state(0),
            _state(0),
            mx.zeros((1, 1, 1)),
            mx.zeros((1, 1, 1)),
            latent_shape=(1, 1, 2),
            noise_seed=1,
        )

    assert pipe.dit is None
    assert pipe._loaded is False
