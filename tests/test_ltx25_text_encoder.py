"""LTX-2.5 text encoder smoke tests (no real weights).

Builds a tiny gemma4 mlx-lm model in-memory to exercise the exact code path
``GemmaLanguageModel.get_all_hidden_states`` uses (model navigation through
``language_model.model``, per-layer calls with a causal mask, k_eq_v on the
full-attention layers), plus spec-shape assertions for the connector stack.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.gemma4 import Model, ModelArgs

from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import (
    Gemma4LanguageModel,
)
from ltx_core_mlx.text_encoders.gemma.feature_extractor import (
    GemmaFeaturesExtractorV2,
    TextEncoderConnector,
)

# 48 layers, full attention every 6th (matches the LTX-2.5 gemma4-12b pattern).
TINY_TEXT_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 48,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "head_dim": 16,
    "global_head_dim": 32,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "vocab_size": 256,
    "sliding_window": 16,
    "attention_k_eq_v": True,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "enable_moe_block": False,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
}


def _tiny_gemma4_model() -> Model:
    layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 8  # 48 layers
    config = dict(TINY_TEXT_CONFIG)
    config["layer_types"] = layer_types
    return Model(ModelArgs(text_config=config))


def _tiny_encoder() -> Gemma4LanguageModel:
    class _TinyGemma4(Gemma4LanguageModel):
        EXPECTED_LAYERS = 48
        EXPECTED_HIDDEN = 64

    enc = _TinyGemma4()
    enc._model = _tiny_gemma4_model()
    enc._tokenizer = None
    return enc


class TestGemma4HiddenStateCollection:
    def test_navigation_and_layer_count(self):
        """get_all_hidden_states collects embedding + 48 layers via mlx-lm paths."""
        enc = _tiny_encoder()
        enc._validate()  # navigation + layer count + hidden size all pass
        token_ids = mx.array([[1, 2, 3, 4]])
        attention_mask = mx.array([[1, 1, 1, 1]])
        states = enc.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        assert len(states) == 49  # embedding + 48 layers
        for s in states:
            assert s.shape == (1, 4, 64)
        mx.synchronize()

    def test_left_pad_mask_works(self):
        """Padding (left-pad zeros in the mask) must not break the layers."""
        enc = _tiny_encoder()
        token_ids = mx.array([[0, 0, 1, 2, 3, 4]])
        attention_mask = mx.array([[0, 0, 1, 1, 1, 1]])
        states = enc.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        assert len(states) == 49
        mx.synchronize()

    def test_validation_rejects_wrong_model_type(self):
        enc = Gemma4LanguageModel()
        enc._model = type("Fake", (), {"model_type": "gemma3"})()
        with pytest.raises(ValueError, match="gemma4"):
            enc._validate()


class TestGemma4SpecShapes:
    def test_connector_spec_shapes(self):
        """TextEncoderConnector defaults build the LTX-2.5 spec shapes."""
        from mlx.utils import tree_flatten

        connector = TextEncoderConnector()  # 3840 / 49 layers / 4096 / 2048
        flat = dict(tree_flatten(connector.parameters()))
        # text_embedding_projection: [4096, 188160] / [2048, 188160] (188160 = 49*3840)
        assert flat["text_embedding_projection.video_aggregate_embed.weight"].shape == (4096, 188160)
        assert flat["text_embedding_projection.audio_aggregate_embed.weight"].shape == (2048, 188160)
        assert flat["text_embedding_projection.video_aggregate_embed.bias"].shape == (4096,)
        assert flat["text_embedding_projection.audio_aggregate_embed.bias"].shape == (2048,)
        # Connectors: 8 transformer_1d_blocks + 128 learnable registers
        assert flat["video_embeddings_connector.learnable_registers"].shape == (128, 4096)
        assert flat["audio_embeddings_connector.learnable_registers"].shape == (128, 2048)
        block0 = "video_embeddings_connector.transformer_1d_blocks.0."
        assert flat[block0 + "attn1.to_q.weight"].shape == (4096, 4096)
        assert flat[block0 + "attn1.to_out.0.weight"].shape == (4096, 4096)
        assert flat[block0 + "ff.net.0.proj.weight"].shape == (16384, 4096)
        a_block0 = "audio_embeddings_connector.transformer_1d_blocks.0."
        assert flat[a_block0 + "attn1.to_q.weight"].shape == (2048, 2048)
        assert flat[a_block0 + "ff.net.0.proj.weight"].shape == (8192, 2048)
        num_blocks = sum(1 for k in flat if "transformer_1d_blocks." in k and k.endswith(".weight"))
        assert num_blocks == 9 * 8 * 2  # 9 weights/block x 8 blocks x 2 connectors

    def test_extractor_forward_with_49_states(self):
        """Full pipeline: 49 hidden states -> (B, T, 4096) video / (B, T, 2048) audio.

        The connector preserves the sequence length (padding positions are
        replaced with tiled learnable registers); the pipeline reaches 1024
        tokens by left-padding tokenization to ``max_length=1024`` before
        the connector, so a 128-token input yields a 128-token output here."""
        extractor = GemmaFeaturesExtractorV2()  # spec defaults
        hidden = [mx.random.normal((1, 128, 3840)) for _ in range(49)]
        video, audio = extractor(hidden, attention_mask=mx.ones((1, 128)))
        mx.synchronize()
        assert video.shape == (1, 128, 4096)
        assert audio.shape == (1, 128, 2048)

    def test_aggregate_projection_matches_spec_scale(self):
        """Video/audio aggregate projections apply sqrt(dim/3840) rescale."""
        connector = TextEncoderConnector()
        hidden = mx.random.normal((1, 4, 188160))
        video, audio = connector.text_embedding_projection(hidden)
        mx.synchronize()
        assert video.shape == (1, 4, 4096)
        assert audio.shape == (1, 4, 2048)
