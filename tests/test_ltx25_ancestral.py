"""LTX-2.5 ancestral (SDE) Euler sampler: stepper math, loop semantics, pipeline wiring.

Covers the port of the official ``EulerAncestralDiffusionStep`` +
``euler_ancestral_denoising_loop`` (eta=1.0, s_noise=1.0, noise seeded from
``seed + 10000``, f32 math with bf16 latent state) and the distilled-pipeline
stage-1 selection: ancestral for ``model_version >= 2.5``, deterministic Euler
for 2.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.utils.samplers import (
    ancestral_denoise_loop,
    ancestral_euler_step,
)


def _reference_ancestral_step(x, d, sigma, sigma_next, noise, eta=1.0, s_noise=1.0):
    """Pure numpy implementation of the official EulerAncestralDiffusionStep.

    Rectified-flow parameterization (alpha = 1 - sigma), exactly as in
    ``ltx_core/components/diffusion_steps.py``.
    """
    if sigma_next == 0:
        return d.astype(np.float32)
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * d
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    renoise_coeff = np.sqrt(max(sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0))
    x_next = (alpha_next / alpha_down) * x_next + noise * s_noise * renoise_coeff
    return x_next


# ---------------------------------------------------------------------------
# ancestral_euler_step — stepper math
# ---------------------------------------------------------------------------


class TestAncestralEulerStep:
    def test_matches_numpy_reference(self):
        """Synthetic step with known sigma/sigma_next reproduces the reference."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((1, 16, 8)).astype(np.float32)
        d = rng.standard_normal((1, 16, 8)).astype(np.float32)
        noise = rng.standard_normal((1, 16, 8)).astype(np.float32)
        # Consecutive DISTILLED_SIGMAS pair (2.5 distilled schedule).
        sigma, sigma_next = 0.975, 0.909375
        expected = _reference_ancestral_step(x, d, sigma, sigma_next, noise, eta=1.0, s_noise=1.0)
        result = ancestral_euler_step(
            mx.array(x), mx.array(d), sigma, sigma_next, mx.array(noise), eta=1.0, s_noise=1.0
        )
        assert result.dtype == mx.float32
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)

    def test_known_sigma_up_sigma_down(self):
        """sigma_down and the injected-noise (sigma_up) coefficient are exact."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 4, 4)).astype(np.float32)
        d = rng.standard_normal((1, 4, 4)).astype(np.float32)
        noise = rng.standard_normal((1, 4, 4)).astype(np.float32)
        sigma, sigma_next = 0.975, 0.909375
        eta, s_noise = 1.0, 1.0

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        sigma_up = np.sqrt(max(sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0))

        # Closed-form manual computation, no helper reuse.
        sigma_down_ratio = sigma_down / sigma
        manual = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * d
        manual = (alpha_next / alpha_down) * manual + noise * s_noise * sigma_up

        result = ancestral_euler_step(mx.array(x), mx.array(d), sigma, sigma_next, mx.array(noise))
        np.testing.assert_allclose(np.array(result), manual, rtol=1e-5, atol=1e-5)

    def test_eta0_reduces_to_plain_euler(self):
        """eta=0 ⇒ sigma_down == sigma_next ⇒ plain Euler interpolation, no noise."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal((1, 8, 4)).astype(np.float32)
        d = rng.standard_normal((1, 8, 4)).astype(np.float32)
        sigma, sigma_next = 0.975, 0.909375
        expected = (sigma_next / sigma) * x + (1.0 - sigma_next / sigma) * d
        result = ancestral_euler_step(mx.array(x), mx.array(d), sigma, sigma_next, None, eta=0.0)
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)

    def test_terminal_sigma_returns_denoised(self):
        x = mx.ones((1, 4, 8)) * 2.0
        d = mx.ones((1, 4, 8)) * -0.5
        result = ancestral_euler_step(x, d, 0.421875, 0.0, mx.zeros_like(x), eta=1.0)
        np.testing.assert_allclose(np.array(result), np.full((1, 4, 8), -0.5), atol=1e-6)

    def test_noise_required_when_eta_gt_0(self):
        with pytest.raises(ValueError, match="noise tensor"):
            ancestral_euler_step(mx.zeros((1, 4, 4)), mx.zeros((1, 4, 4)), 1.0, 0.5, None, eta=1.0)

    def test_bf16_inputs_step_in_float32(self):
        """bf16 latents are cast to f32 for the step math (reference verbatim)."""
        rng = np.random.default_rng(3)
        x = mx.array(rng.standard_normal((1, 8, 4)).astype(np.float32), dtype=mx.bfloat16)
        d = mx.array(rng.standard_normal((1, 8, 4)).astype(np.float32), dtype=mx.bfloat16)
        noise = mx.array(rng.standard_normal((1, 8, 4)).astype(np.float32), dtype=mx.bfloat16)
        result = ancestral_euler_step(x, d, 0.975, 0.909375, noise, eta=1.0, s_noise=1.0)
        assert result.dtype == mx.float32
        expected = _reference_ancestral_step(
            np.array(x.astype(mx.float32)),
            np.array(d.astype(mx.float32)),
            0.975,
            0.909375,
            np.array(noise.astype(mx.float32)),
        )
        np.testing.assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# ancestral_denoise_loop — loop semantics
# ---------------------------------------------------------------------------


class _FakeModel:
    """Deterministic fake X0Model: records calls, returns x0 = 0.5 * input."""

    def __init__(self):
        self.calls = 0
        self.input_dtypes: list[tuple] = []
        self.last_video_x0: mx.array | None = None
        self.last_audio_x0: mx.array | None = None

    def __call__(self, video_latent, audio_latent, **kwargs):
        self.calls += 1
        self.input_dtypes.append((video_latent.dtype, audio_latent.dtype))
        video_x0 = (0.5 * video_latent).astype(mx.bfloat16)
        audio_x0 = (0.5 * audio_latent).astype(mx.bfloat16)
        self.last_video_x0 = video_x0
        self.last_audio_x0 = audio_x0
        return video_x0, audio_x0


def _state(shape: tuple, seed: int) -> LatentState:
    """Build a bf16 latent state with seeded Gaussian noise (like stage 1)."""
    mx.random.seed(seed)
    latent = mx.random.normal(shape).astype(mx.bfloat16)
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros(shape, dtype=mx.bfloat16),
        denoise_mask=mx.ones((shape[0], shape[1], 1), dtype=mx.bfloat16),
    )


def _run_loop(
    seed,
    noise_seed,
    sigmas=None,
    eta=1.0,
    s_noise=1.0,
    model=None,
    noise_step_indices=None,
    noise_total_steps=None,
):
    """Run ancestral_denoise_loop with fake model + tiny states; return (out, model)."""
    video_state = _state((1, 32, 8), seed)
    audio_state = _state((1, 16, 8), seed + 1)
    fake = model or _FakeModel()
    out = ancestral_denoise_loop(
        model=fake,
        video_state=video_state,
        audio_state=audio_state,
        video_text_embeds=mx.zeros((1, 4, 8)),
        audio_text_embeds=mx.zeros((1, 4, 8)),
        sigmas=sigmas if sigmas is not None else list(DISTILLED_SIGMAS),
        show_progress=False,
        noise_seed=noise_seed,
        noise_step_indices=noise_step_indices,
        noise_total_steps=noise_total_steps,
        eta=eta,
        s_noise=s_noise,
    )
    mx.eval(out.video_latent, out.audio_latent)
    return out, fake


class TestAncestralDenoiseLoop:
    def test_explicit_identity_noise_mapping_preserves_default_output(self):
        sigmas = [1.0, 0.75, 0.5, 0.0]
        default, _ = _run_loop(42, 10042, sigmas=sigmas)
        mapped, _ = _run_loop(
            42,
            10042,
            sigmas=sigmas,
            noise_step_indices=[0, 1, 2],
            noise_total_steps=3,
        )
        assert mx.array_equal(default.video_latent, mapped.video_latent).item()
        assert mx.array_equal(default.audio_latent, mapped.audio_latent).item()

    def test_compressed_schedule_selects_original_noise_lanes(self, monkeypatch):
        import ltx_pipelines_mlx.utils.samplers as samplers

        noise_seed = 10042
        selected_keys = []
        original_helper = samplers._ancestral_noise_from_key

        def record_key(step_key, video_shape, audio_shape):
            selected_keys.append(np.array(step_key))
            return original_helper(step_key, video_shape, audio_shape)

        monkeypatch.setattr(samplers, "_ancestral_noise_from_key", record_key)
        _run_loop(
            42,
            noise_seed,
            sigmas=[1.0, 0.5, 0.0],
            noise_step_indices=[3, 7],
            noise_total_steps=8,
        )

        # The terminal transition does not draw noise; the non-terminal one
        # must use lane 3 from the original eight-transition stream.
        expected_key = mx.random.split(mx.random.key(noise_seed), 8)[3]
        assert len(selected_keys) == 1
        np.testing.assert_array_equal(selected_keys[0], np.array(expected_key))

    @pytest.mark.parametrize(
        ("indices", "total", "message"),
        [
            ([0, 1], None, "provided together"),
            (None, 2, "provided together"),
            ([0, 1], 0, "must be positive"),
            ([0], 2, "one index per transition"),
            ([0, 2], 2, "must be in"),
        ],
    )
    def test_noise_mapping_validation(self, indices, total, message):
        with pytest.raises(ValueError, match=message):
            _run_loop(
                42,
                10042,
                sigmas=[1.0, 0.5, 0.0],
                noise_step_indices=indices,
                noise_total_steps=total,
            )

    def test_same_seed_is_deterministic(self):
        out1, fake1 = _run_loop(42, 42 + 10000)
        out2, fake2 = _run_loop(42, 42 + 10000)
        assert fake1.calls == fake2.calls == len(DISTILLED_SIGMAS) - 1 == 8
        assert mx.array_equal(out1.video_latent, out2.video_latent).item()
        assert mx.array_equal(out1.audio_latent, out2.audio_latent).item()

    def test_different_noise_seeds_differ(self):
        out1, _ = _run_loop(42, 42 + 10000)
        out2, _ = _run_loop(42, 43 + 10000)
        assert not mx.array_equal(out1.video_latent, out2.video_latent).item()
        assert not mx.array_equal(out1.audio_latent, out2.audio_latent).item()

    def test_different_pipeline_seeds_differ(self):
        out1, _ = _run_loop(42, 42 + 10000)
        out2, _ = _run_loop(43, 43 + 10000)
        assert not mx.array_equal(out1.video_latent, out2.video_latent).item()

    def test_terminal_step_returns_denoised_prediction(self):
        """Last sigma 0.0 ⇒ output equals the model's x0 (no renoise)."""
        out, fake = _run_loop(7, 7 + 10000, sigmas=[1.0, 0.5, 0.0])
        assert fake.last_video_x0 is not None and fake.last_audio_x0 is not None
        assert mx.array_equal(out.video_latent, fake.last_video_x0).item()
        assert mx.array_equal(out.audio_latent, fake.last_audio_x0).item()

    def test_eta0_ignores_noise_seed(self):
        """eta=0 ⇒ no noise drawn ⇒ noise_seed has no effect (reference gate)."""
        out1, fake1 = _run_loop(42, 1, eta=0.0)
        out2, fake2 = _run_loop(42, 2, eta=0.0)
        assert fake1.calls == fake2.calls
        assert mx.array_equal(out1.video_latent, out2.video_latent).item()
        assert mx.array_equal(out1.audio_latent, out2.audio_latent).item()

    def test_model_called_with_bf16_latents(self):
        """Latent state is bf16; model inputs and outputs stay bf16 (f32 step only)."""
        out, fake = _run_loop(1, 1 + 10000, sigmas=[1.0, 0.5, 0.0])
        for video_dtype, audio_dtype in fake.input_dtypes:
            assert video_dtype == mx.bfloat16
            assert audio_dtype == mx.bfloat16
        assert out.video_latent.dtype == mx.bfloat16
        assert out.audio_latent.dtype == mx.bfloat16

    def test_step_callback_observes_every_completed_transition(self):
        observed = []
        video_state = _state((1, 8, 4), 12)
        audio_state = _state((1, 4, 4), 13)
        out = ancestral_denoise_loop(
            model=_FakeModel(),
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((1, 2, 4)),
            audio_text_embeds=mx.zeros((1, 2, 4)),
            sigmas=[1.0, 0.5, 0.0],
            show_progress=False,
            noise_seed=10012,
            step_callback=lambda sigma, video, audio: observed.append((sigma, video, audio)),
        )
        mx.eval(out.video_latent, out.audio_latent)

        assert [item[0] for item in observed] == [0.5, 0.0]
        assert mx.array_equal(observed[-1][1], out.video_latent).item()
        assert mx.array_equal(observed[-1][2], out.audio_latent).item()


# ---------------------------------------------------------------------------
# DistilledPipeline stage-1 sampler selection
# ---------------------------------------------------------------------------


def _write_config(path, version: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if version.startswith("2.5"):
        (path / "embedded_config.json").write_text(
            json.dumps({"transformer": {"use_keyframes_abs_pos_embedding": True}})
        )
    else:
        (path / "config.json").write_text(json.dumps({"model_version": "2.3.0"}))


class _FakeVAEEncoder:
    def denormalize_latent(self, x):
        return x

    def normalize_latent(self, x):
        return x

    def encode(self, x):
        return x


class _FakeUpsampler:
    def __call__(self, x):
        # (1, C, F, H, W) -> (1, C, F, 2H, 2W) — spatial x2 upscale.
        return mx.zeros(
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
            dtype=x.dtype,
        )


class TestDistilledPipelineSamplerSelection:
    def test_should_use_ancestral_sampler(self, tmp_path):
        from ltx_pipelines_mlx.distilled import ANCESTRAL_ETA, ANCESTRAL_S_NOISE, should_use_ancestral_sampler

        assert ANCESTRAL_ETA == 1.0
        assert ANCESTRAL_S_NOISE == 1.0
        _write_config(tmp_path / "m25", "2.5.0")
        _write_config(tmp_path / "m23", "2.3.0")
        assert should_use_ancestral_sampler(tmp_path / "m25") is True
        assert should_use_ancestral_sampler(tmp_path / "m23") is False

    def test_stage1_ancestral_for_25_deterministic_for_23(self, tmp_path, monkeypatch):
        """Stage 1 uses the ancestral loop for >=2.5, deterministic for 2.3;
        stage 2 is always deterministic."""
        from ltx_pipelines_mlx.distilled import DistilledPipeline
        from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

        calls = {"ancestral": 0, "deterministic": 0}
        observed_noise_seed = []

        def fake_ancestral(model, video_state, audio_state, **kwargs):
            calls["ancestral"] += 1
            observed_noise_seed.append(kwargs["noise_seed"])
            assert kwargs["eta"] == 1.0 and kwargs["s_noise"] == 1.0
            return DenoiseOutput(video_latent=video_state.latent, audio_latent=audio_state.latent)

        def fake_deterministic(model, video_state, audio_state, **kwargs):
            calls["deterministic"] += 1
            return DenoiseOutput(video_latent=video_state.latent, audio_latent=audio_state.latent)

        import ltx_pipelines_mlx.distilled as distilled_mod

        monkeypatch.setattr(distilled_mod, "ancestral_denoise_loop", fake_ancestral)
        monkeypatch.setattr(distilled_mod, "denoise_loop", fake_deterministic)

        for version, expect_ancestral in (("2.5.0", True), ("2.3.0", False)):
            calls["ancestral"] = calls["deterministic"] = 0
            observed_noise_seed.clear()

            model_dir = tmp_path / version
            _write_config(model_dir, version)
            pipe = DistilledPipeline(str(model_dir), low_memory=False, low_ram_streaming=False)
            pipe.verbose = False
            assert pipe.use_ancestral_sampler is expect_ancestral

            # Stub the heavy machinery; keep the real stage-1/2 wiring.
            pipe._load_text_encoder = lambda: None
            pipe._encode_text = lambda prompt: (mx.zeros((1, 2, 8)), mx.zeros((1, 2, 8)))
            pipe.load = lambda: None
            pipe.dit = object()
            pipe.vae_encoder = _FakeVAEEncoder()
            pipe.upsampler = _FakeUpsampler()

            pipe.generate_two_stage(
                "a cat on a mat",
                height=32,
                width=32,
                num_frames=5,
                frame_rate=24.0,
                seed=42,
            )

            if expect_ancestral:
                assert calls["ancestral"] == 1, f"{version}: stage 1 must use ancestral"
                assert calls["deterministic"] == 1, f"{version}: stage 2 stays deterministic"
                assert observed_noise_seed == [42 + 10000], "noise_seed must be seed + 10000"
            else:
                assert calls["ancestral"] == 0, f"{version}: 2.3 must not use ancestral"
                assert calls["deterministic"] == 2, f"{version}: both stages deterministic"

    def test_fast_stage1_uses_qualified_schedule_and_original_noise_lanes(self, tmp_path, monkeypatch):
        from ltx_pipelines_mlx.distilled import DistilledPipeline
        from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

        observed = {}

        def fake_ancestral(model, video_state, audio_state, **kwargs):
            observed.update(kwargs)
            return DenoiseOutput(video_latent=video_state.latent, audio_latent=audio_state.latent)

        def fake_deterministic(model, video_state, audio_state, **kwargs):
            return DenoiseOutput(video_latent=video_state.latent, audio_latent=audio_state.latent)

        import ltx_pipelines_mlx.distilled as distilled_mod

        monkeypatch.setattr(distilled_mod, "ancestral_denoise_loop", fake_ancestral)
        monkeypatch.setattr(distilled_mod, "denoise_loop", fake_deterministic)
        monkeypatch.setattr(distilled_mod, "aggressive_cleanup", lambda: None)

        model_dir = tmp_path / "m25"
        _write_config(model_dir, "2.5.0")
        pipe = DistilledPipeline(str(model_dir), low_memory=False, low_ram_streaming=False)
        pipe.verbose = False
        pipe._fast_stage1_package = SimpleNamespace(
            transformer_path=Path("/model/transformer-distilled.safetensors"),
            contract=SimpleNamespace(
                schedule=(1.0, 0.98125, 0.909375, 0.421875, 0.0),
                noise_step_indices=(0, 3, 5, 7),
                noise_total_steps=8,
            ),
        )
        pipe._load_text_encoder = lambda: None
        pipe._encode_text = lambda prompt: (mx.zeros((1, 2, 8)), mx.zeros((1, 2, 8)))
        pipe.load = lambda: None
        pipe._load_transformer_with_optional_streaming = lambda _path: object()
        pipe.dit = object()
        pipe.vae_encoder = _FakeVAEEncoder()
        pipe.upsampler = _FakeUpsampler()

        pipe.generate_two_stage(
            "a cat on a mat",
            height=32,
            width=32,
            num_frames=5,
            frame_rate=24.0,
            seed=42,
        )

        assert observed["sigmas"] == [1.0, 0.98125, 0.909375, 0.421875, 0.0]
        assert observed["noise_step_indices"] == [0, 3, 5, 7]
        assert observed["noise_total_steps"] == 8
        assert pipe.dit is None
        assert pipe._loaded is False
