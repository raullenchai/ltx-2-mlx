"""LTX-2.5 transformer deltas: keyframes_abs_pos_embedding + config mapping.

No real weights: models are built with tiny configs and random arrays.
"""

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_core_mlx.utils.positions import compute_keyframes_mask


def _tiny_config(**overrides) -> LTXModelConfig:
    base = dict(
        num_layers=2,
        video_dim=32,
        audio_dim=16,
        video_num_heads=4,
        audio_num_heads=4,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=4,
        av_cross_head_dim=4,
        video_patch_channels=8,
        audio_patch_channels=8,
        ff_mult=2.0,
        timestep_embedding_dim=32,
    )
    base.update(overrides)
    return LTXModelConfig(**base)


class TestKeyframesConfig:
    def test_default_off_for_23(self):
        """2.3 compatibility: flag defaults to False, no parameter registered."""
        model = LTXModel(_tiny_config())
        assert model.config.use_keyframes_abs_pos_embedding is False
        assert getattr(model, "keyframes_abs_pos_embedding", None) is None
        keys = {k for k, _ in model.parameters().items()}
        assert not any("keyframes" in k for k in keys)

    def test_flag_registers_parameter(self):
        """2.5: flag True registers keyframes_abs_pos_embedding (1, video_dim)."""
        model = LTXModel(_tiny_config(use_keyframes_abs_pos_embedding=True))
        emb = model.keyframes_abs_pos_embedding
        assert emb is not None
        assert emb.shape == (1, 32)  # (1, video_dim)

    def test_from_checkpoint_config_maps_flag(self):
        """from_checkpoint_config reads the 2.5 flag (and model_version)."""
        cfg = LTXModelConfig.from_checkpoint_config(
            {
                "transformer": {
                    "num_layers": 48,
                    "cross_attention_dim": 4096,
                    "audio_cross_attention_dim": 2048,
                    "use_keyframes_abs_pos_embedding": True,
                    "model_version": "2.5.0",
                }
            }
        )
        assert cfg.use_keyframes_abs_pos_embedding is True
        assert cfg.model_version == "2.5.0"

    def test_from_checkpoint_config_default_off(self):
        """2.3 checkpoint config (no flag) keeps the default False."""
        cfg = LTXModelConfig.from_checkpoint_config({"transformer": {"num_layers": 48}})
        assert cfg.use_keyframes_abs_pos_embedding is False
        assert cfg.model_version == "2.3"


class TestKeyframesForward:
    def _model(self):
        return LTXModel(_tiny_config(use_keyframes_abs_pos_embedding=True))

    def _inputs(self, nv=8, na=4, nt=3):
        return dict(
            video_latent=mx.random.normal((1, nv, 8)),
            audio_latent=mx.random.normal((1, na, 8)),
            timestep=mx.array([0.5]),
            video_text_embeds=mx.zeros((1, nt, 32)),
            audio_text_embeds=mx.zeros((1, nt, 16)),
        )

    def test_forward_with_mask_shapes(self):
        """Forward with video_keyframes_mask keeps output shapes."""
        model = self._model()
        mx.random.seed(0)
        v_out, a_out = model(**self._inputs(), video_keyframes_mask=mx.zeros((1, 8)))
        mx.synchronize()
        assert v_out.shape == (1, 8, 8)
        assert a_out.shape == (1, 4, 8)

    def test_zero_mask_equals_no_mask(self):
        """A zero mask is an exact no-op (mask * embedding = 0)."""
        model = self._model()
        mx.random.seed(7)
        out_plain_v, out_plain_a = model(**self._inputs())
        mx.random.seed(7)
        out_masked_v, out_masked_a = model(
            **self._inputs(), video_keyframes_mask=mx.zeros((1, 8))
        )
        mx.synchronize()
        assert mx.allclose(out_plain_v, out_masked_v).item()
        assert mx.allclose(out_plain_a, out_masked_a).item()

    def test_masked_tokens_receive_embedding(self):
        """Marked tokens get the embedding added right after patchify_proj."""
        model = self._model()
        # Deterministic, non-trivial embedding.
        emb = mx.arange(32, dtype=mx.float32).reshape(1, 32) / 32.0
        model.keyframes_abs_pos_embedding = emb
        # Unit check of the add itself: with a zero input the patched hidden
        # is the projection bias, and marked positions get + embedding.
        hidden = model.patchify_proj(mx.zeros((1, 8, 8)))
        mask = mx.zeros((1, 8))
        mask[:, :2] = 1.0
        added = model._apply_keyframes_abs_pos_embedding(hidden, mask)
        mx.synchronize()
        # Unmarked tokens unchanged; marked tokens differ by exactly the embedding.
        assert mx.allclose(added[:, 0, :], hidden[:, 0, :] + emb[0]).item()
        assert mx.allclose(added[:, 1, :], hidden[:, 1, :] + emb[0]).item()
        assert mx.allclose(added[:, 2:, :], hidden[:, 2:, :]).item()

    def test_masked_forward_differs_from_unmasked(self):
        """End-to-end: a non-zero mask changes the forward (via attention mixing)."""
        model = self._model()
        emb = mx.arange(32, dtype=mx.float32).reshape(1, 32) / 32.0
        model.keyframes_abs_pos_embedding = emb
        inputs = self._inputs()
        inputs["video_latent"] = mx.zeros((1, 8, 8))
        mask = mx.zeros((1, 8))
        mask[:, :2] = 1.0
        v_out, _ = model(**inputs, video_keyframes_mask=mask)
        v_plain, _ = model(**inputs, video_keyframes_mask=mx.zeros((1, 8)))
        mx.synchronize()
        assert not mx.allclose(v_out, v_plain).item()

    def test_23_model_ignores_mask(self):
        """2.3 model (no parameter) ignores the mask entirely (no-op)."""
        model = LTXModel(_tiny_config())
        mx.random.seed(3)
        out_v, out_a = model(**self._inputs(), video_keyframes_mask=mx.ones((1, 8)))
        mx.random.seed(3)
        plain_v, plain_a = model(**self._inputs())
        mx.synchronize()
        assert mx.allclose(out_v, plain_v).item()
        assert mx.allclose(out_a, plain_a).item()

    def test_x0_wrapper_infers_mask_from_global_positions(self):
        """Production X0 calls infer the first-frame mask before optional tiling."""
        model = self._model()
        wrapper = X0Model(model)
        inputs = self._inputs()
        sigma = inputs.pop("timestep")
        # Four tokens share the first latent-frame midpoint; the remainder
        # belong to later temporal positions.
        video_positions = mx.array(
            [[[0.1, 0.0, 0.0]] * 4 + [[0.2, 0.0, 0.0]] * 4],
            dtype=mx.float32,
        )
        inputs["video_positions"] = video_positions
        explicit_mask = mx.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        auto_v, auto_a = wrapper(sigma=sigma, **inputs)
        explicit_v, explicit_a = wrapper(
            sigma=sigma,
            video_keyframes_mask=explicit_mask,
            **inputs,
        )
        mx.synchronize()
        assert mx.allclose(auto_v, explicit_v).item()
        assert mx.allclose(auto_a, explicit_a).item()

    def test_gate_signal_matches_forward_prelude(self):
        """compute_gate_signal with the mask equals block 0's modulated input."""
        model = self._model()
        mask = mx.zeros((1, 8))
        mask[:, 0] = 1.0
        B, Nv, Na = 1, 8, 4
        video_latent = mx.random.normal((B, Nv, 8))
        audio_latent = mx.random.normal((B, Na, 8))
        timestep = mx.array([0.5], dtype=mx.bfloat16)

        gate = model.compute_gate_signal(
            video_latent, audio_latent, timestep, video_keyframes_mask=mask
        )
        # Replicate the prelude manually.
        v_hidden = model.patchify_proj(video_latent.astype(mx.bfloat16))
        v_hidden = model._apply_keyframes_abs_pos_embedding(v_hidden, mask)
        t_emb = model._embed_timestep_scalar(timestep)
        video_adaln_emb, _ = model.adaln_single(t_emb)
        ref = model.transformer_blocks[0].compute_video_normed_sa(v_hidden, video_adaln_emb)
        mx.synchronize()
        assert mx.allclose(gate, ref, atol=1e-5).item()


class TestKeyframesMaskHelper:
    def test_marks_first_latent_frame(self):
        mask = compute_keyframes_mask(num_latent_frames=3, height=2, width=2)
        assert mask.shape == (1, 12, 1)
        assert mask[0, :4, 0].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert mask[0, 4:, 0].tolist() == [0.0] * 8

    def test_appended_tokens_unmarked(self):
        mask = compute_keyframes_mask(num_latent_frames=2, height=2, width=2, num_appended_tokens=4)
        assert mask.shape == (1, 12, 1)
        assert mask[0, :4, 0].tolist() == [1.0] * 4
        assert mask[0, 4:, 0].tolist() == [0.0] * 8
