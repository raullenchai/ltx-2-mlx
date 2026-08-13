"""Conversion script smoke tests with synthetic single-files (no real weights).

Builds tiny fake official-format safetensors files and runs every conversion
step end-to-end, asserting the split MLX layout the pipeline expects.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_core_mlx.model.duration_head import DurationHead
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from scripts.convert_ltx25_to_mlx import (
    convert_audio_vae,
    convert_config,
    convert_connector,
    convert_duration_head,
    convert_text_encoder,
    convert_transformer,
    convert_upscalers,
    convert_vae_conv,
    rename_2_3,
)

TINY_TRANSFORMER_CONFIG = {
    "num_layers": 2,
    "cross_attention_dim": 64,
    "audio_cross_attention_dim": 64,
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "audio_num_attention_heads": 4,
    "audio_attention_head_dim": 16,
    "in_channels": 8,
    "audio_in_channels": 8,
    "out_channels": 8,
    "audio_out_channels": 8,
    "ff_mult": 2.0,
    "timestep_embedding_dim": 32,
    "positional_embedding_max_pos": [20, 64, 64],
    "audio_positional_embedding_max_pos": [20],
    "use_keyframes_abs_pos_embedding": True,
}


def _tiny_dit():
    return LTXModel(LTXModelConfig.from_checkpoint_config(TINY_TRANSFORMER_CONFIG))


def _make_transformer_single_file(path, dit, metadata=None):
    """Write a fake official single-file from a tiny DiT + fake connector tensors."""
    from mlx.utils import tree_flatten

    weights = {}
    for k, v in tree_flatten(dit.parameters()):
        weights[f"model.diffusion_model.{k}"] = np.array(v, dtype=np.float32)
    # Fake connector tensors (real shapes would come from the same file).
    weights["model.diffusion_model.video_embeddings_connector.learnable_registers"] = np.zeros((8, 64), dtype=np.float32)
    weights["model.diffusion_model.audio_embeddings_connector.learnable_registers"] = np.zeros((8, 64), dtype=np.float32)
    meta = (
        json.dumps({"transformer": TINY_TRANSFORMER_CONFIG, "model_version": "2.5.0"})
        if metadata is None
        else json.dumps(metadata)
    )
    save_file(weights, str(path), metadata={"config": meta})


def _make_te_file(path, text_config):
    """Write a fake gemma4 TE single-file from a tiny mlx-lm gemma4 model."""
    from mlx_lm.models.gemma4 import Model, ModelArgs

    model = Model(ModelArgs(text_config=text_config))
    from mlx.utils import tree_flatten

    weights = {}
    for k, v in tree_flatten(model.parameters()):
        # model.* layout (strip language_model.model.)
        assert k.startswith("language_model.model.")
        weights["model." + k[len("language_model.model.") :]] = np.array(v, dtype=np.float32)
    weights["text_embedding_projection.video_aggregate_embed.weight"] = np.zeros((32, 188160), dtype=np.float32)
    weights["text_embedding_projection.video_aggregate_embed.bias"] = np.zeros((32,), dtype=np.float32)
    weights["text_embedding_projection.audio_aggregate_embed.weight"] = np.zeros((16, 188160), dtype=np.float32)
    weights["text_embedding_projection.audio_aggregate_embed.bias"] = np.zeros((16,), dtype=np.float32)
    weights["tokenizer_json"] = np.frombuffer(b'{"fake": "tokenizer"}', dtype=np.uint8)
    weights["hf_asset__tokenizer_config.json"] = np.frombuffer(b'{"pad_token": "<pad>"}', dtype=np.uint8)
    weights["hf_asset__generation_config.json"] = np.frombuffer(b'{"bos_token_id": 2}', dtype=np.uint8)
    save_file(
        weights,
        str(path),
        metadata={"gemma_config": json.dumps({"text_config": text_config})},
    )
    return model


TINY_GEMMA4_TEXT_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
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
    "layer_types": ["sliding_attention", "full_attention"],
}


class TestRename:
    def test_linear_1_to_linear1(self):
        assert rename_2_3("adaln_single.emb.timestep_embedder.linear_1.weight") == (
            "adaln_single.emb.timestep_embedder.linear1.weight"
        )
        assert rename_2_3("adaln_single.emb.timestep_embedder.linear_2.weight") == (
            "adaln_single.emb.timestep_embedder.linear2.weight"
        )
        # Official 2.5 single-files use Comfy-style names; the MLX blocks use
        # the flat 2.3 forms.
        assert rename_2_3("transformer_blocks.0.ff.net.0.proj.weight") == (
            "transformer_blocks.0.ff.proj_in.weight"
        )
        assert rename_2_3("transformer_blocks.0.ff.net.2.weight") == (
            "transformer_blocks.0.ff.proj_out.weight"
        )
        assert rename_2_3("transformer_blocks.0.audio_ff.net.0.proj.bias") == (
            "transformer_blocks.0.audio_ff.proj_in.bias"
        )
        assert rename_2_3("transformer_blocks.0.attn1.to_out.0.weight") == (
            "transformer_blocks.0.attn1.to_out.weight"
        )


class TestConvertConfig:
    def test_writes_flat_configs(self, tmp_path):
        src = tmp_path / "transformer.safetensors"
        _make_transformer_single_file(src, _tiny_dit())
        out = tmp_path / "out"
        convert_config(out, src)
        flat = json.loads((out / "config.json").read_text())
        assert flat["model_version"] == "2.5.0"
        assert flat["use_keyframes_abs_pos_embedding"] is True
        assert flat["num_layers"] == 2
        embedded = json.loads((out / "embedded_config.json").read_text())
        assert embedded["transformer"]["cross_attention_dim"] == 64


class TestConvertTransformer:
    def test_q8_roundtrip_layout(self, tmp_path):
        src = tmp_path / "transformer.safetensors"
        _make_transformer_single_file(src, _tiny_dit())
        out = tmp_path / "out"
        convert_transformer(src, out, "transformer-distilled.safetensors", q8=True)

        import mlx.core as mx

        raw = mx.load(str(out / "transformer-distilled.safetensors"))
        assert "transformer.keyframes_abs_pos_embedding" in raw
        assert raw["transformer.keyframes_abs_pos_embedding"].shape == (1, 64)
        # No connector keys in the transformer file.
        assert not any("embeddings_connector" in k for k in raw)
        # linear1 naming.
        assert "transformer.adaln_single.emb.timestep_embedder.linear1.weight" in raw
        # Quantized blocks expose .scales keys.
        assert any(
            k.startswith("transformer.transformer_blocks.0.") and k.endswith(".scales")
            for k in raw
        )
        # Every Linear weight under transformer_blocks has a .scales counterpart
        # (RMSNorm q/k_norm weights are not quantized).
        for k in raw:
            if not (k.startswith("transformer.transformer_blocks.") and k.endswith(".weight")):
                continue
            if k.endswith(("q_norm.weight", "k_norm.weight")):
                continue
            assert k[: -len(".weight")] + ".scales" in raw, k

    def test_bf16_roundtrip_layout(self, tmp_path):
        src = tmp_path / "transformer.safetensors"
        _make_transformer_single_file(src, _tiny_dit())
        out = tmp_path / "out"
        convert_transformer(src, out, "transformer-dev.safetensors", q8=False)
        raw = mx.load(str(out / "transformer-dev.safetensors"))
        assert "transformer.keyframes_abs_pos_embedding" in raw
        assert not any(".scales" in k for k in raw)


class TestConvertConnector:
    def test_layout(self, tmp_path):
        tf = tmp_path / "transformer.safetensors"
        _make_transformer_single_file(tf, _tiny_dit())
        te = tmp_path / "te.safetensors"
        _make_te_file(te, TINY_GEMMA4_TEXT_CONFIG)
        out = tmp_path / "out"
        convert_connector(out, tf, te)
        raw = mx.load(str(out / "connector.safetensors"))
        assert "connector.video_embeddings_connector.learnable_registers" in raw
        assert "connector.audio_embeddings_connector.learnable_registers" in raw
        assert "connector.text_embedding_projection.video_aggregate_embed.weight" in raw
        assert "connector.text_embedding_projection.audio_aggregate_embed.weight" in raw


class TestConvertTextEncoder:
    @pytest.fixture(autouse=True)
    def _tiny_te_layers(self, monkeypatch):
        monkeypatch.setenv("LTX25_CONVERT_EXPECTED_TE_LAYERS", "2")

    def test_bf16_layout(self, tmp_path):
        te = tmp_path / "te.safetensors"
        _make_te_file(te, TINY_GEMMA4_TEXT_CONFIG)
        out = tmp_path / "out"
        convert_text_encoder(te, out, gemma4_bits=None)

        te_out = out / "text_encoder"
        config = json.loads((te_out / "config.json").read_text())
        assert config["model_type"] == "gemma4"
        assert config["text_config"]["num_hidden_layers"] == 2
        assert (te_out / "tokenizer.json").read_bytes() == b'{"fake": "tokenizer"}'
        assert (te_out / "tokenizer_config.json").read_bytes() == b'{"pad_token": "<pad>"}'
        assert (te_out / "generation_config.json").read_bytes() == b'{"bos_token_id": 2}'

        raw = mx.load(str(te_out / "model.safetensors"))
        assert "language_model.model.embed_tokens.weight" in raw
        assert any(k.startswith("language_model.model.layers.0.") for k in raw)
        assert not any("text_embedding_projection" in k for k in raw)

    def test_4bit_quantize_roundtrip(self, tmp_path):
        """--gemma4-bits 4: mlx-lm must reload the quantized dir (0.31.3 gemma4)."""
        te = tmp_path / "te.safetensors"
        _make_te_file(te, TINY_GEMMA4_TEXT_CONFIG)
        out = tmp_path / "out"
        convert_text_encoder(te, out, gemma4_bits=4)

        te_out = out / "text_encoder"
        config = json.loads((te_out / "config.json").read_text())
        assert config["quantization"] == {"group_size": 64, "bits": 4, "mode": "affine"}
        raw = mx.load(str(te_out / "model.safetensors"))
        assert any(k.endswith(".scales") for k in raw)

        # Reload through mlx-lm to prove the dir is a loadable checkpoint.
        from mlx_lm.utils import load_model

        model, _ = load_model(te_out, lazy=True)
        assert model.model_type == "gemma4"
        # Sanitize+load worked: embed_tokens holds real (non-init) weights.
        mx.eval(model.parameters())


class TestConvertVaeAudioDurationUpscalers:
    def test_vae_conv_split(self, tmp_path):
        src = tmp_path / "vae_conv.safetensors"
        # Non-trivial real per-channel pair (the official file ships one pair
        # shared by encoder and decoder).
        real_mean = np.linspace(-0.5, 0.5, 4, dtype=np.float32)
        real_std = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        weights = {
            "encoder.conv_in.conv.weight": np.zeros((4, 3, 1, 1, 1), dtype=np.float32),
            "encoder.conv_in.conv.bias": np.zeros((4,), dtype=np.float32),
            "decoder.conv_out.conv.weight": np.zeros((3, 4, 1, 1, 1), dtype=np.float32),
            "decoder.conv_out.conv.bias": np.zeros((3,), dtype=np.float32),
            "per_channel_statistics.mean-of-means": real_mean,
            "per_channel_statistics.std-of-means": real_std,
        }
        save_file(weights, str(src))
        out = tmp_path / "out"
        convert_vae_conv(src, out)
        enc = mx.load(str(out / "vae_encoder.safetensors"))
        dec = mx.load(str(out / "vae_decoder.safetensors"))
        assert "vae_encoder.conv_in.conv.weight" in enc
        assert "vae_encoder.per_channel_statistics._mean_of_means" in enc
        assert "vae_decoder.conv_out.conv.weight" in dec
        # Official 2.5 files ship PyTorch (O, I, D, H, W); the MLX port needs
        # (O, D, H, W, I) — the converter must permute conv weights.
        assert enc["vae_encoder.conv_in.conv.weight"].shape == (4, 1, 1, 1, 3)
        assert dec["vae_decoder.conv_out.conv.weight"].shape == (3, 1, 1, 1, 4)
        # The decoder gets the REAL per-channel pair (same values as the
        # encoder's mean-of-means), NOT the port-1 synthesized identity.
        dec_mean = dec["vae_decoder.per_channel_statistics.mean"]
        dec_std = dec["vae_decoder.per_channel_statistics.std"]
        assert dec_mean.shape == (4,)
        assert dec_std.shape == (4,)
        assert mx.allclose(dec_mean.astype(mx.float32), mx.array(real_mean)).item()
        assert mx.allclose(dec_std.astype(mx.float32), mx.array(real_std)).item()
        assert not mx.equal(dec_mean.astype(mx.float32), mx.zeros((4,))).all().item()
        assert not mx.equal(dec_std.astype(mx.float32), mx.ones((4,))).all().item()

    def test_vae_conv_split_missing_stats_fallback(self, tmp_path):
        """No stats anywhere: identity fallback + warning (last resort only)."""
        src = tmp_path / "vae_conv.safetensors"
        weights = {
            "encoder.conv_in.conv.weight": np.zeros((4, 3, 1, 1, 1), dtype=np.float32),
            "decoder.conv_out.conv.weight": np.zeros((3, 4, 1, 1, 1), dtype=np.float32),
        }
        save_file(weights, str(src))
        out = tmp_path / "out"
        convert_vae_conv(src, out)
        dec = mx.load(str(out / "vae_decoder.safetensors"))
        # Identity fallback keeps the runtime loadable (128 channels).
        assert dec["vae_decoder.per_channel_statistics.mean"].shape == (128,)
        assert dec["vae_decoder.per_channel_statistics.std"].shape == (128,)

    def test_audio_vae_split(self, tmp_path):
        src = tmp_path / "audio.safetensors"
        weights = {
            "audio_vae.encoder.conv_in.conv.weight": np.zeros((4, 2, 3), dtype=np.float32),
            "audio_vae.decoder.conv_out.conv.weight": np.zeros((2, 4, 3, 3), dtype=np.float32),
            "audio_vae.per_channel_statistics.mean-of-means": np.zeros((4,), dtype=np.float32),
            "vocoder.vocoder.resblocks.0.convs.0.weight": np.zeros((2, 2, 3), dtype=np.float32),
            "vocoder.vocoder.ups.0.weight": np.zeros((6, 4, 5), dtype=np.float32),
            "vocoder.vocoder.act_post.upsample.filter": np.zeros((1, 1, 12), dtype=np.float32),
            "vocoder.mel_stft.stft_fn.forward_basis": np.zeros((4, 1, 3), dtype=np.float32),
        }
        save_file(weights, str(src))
        out = tmp_path / "out"
        convert_audio_vae(src, out)
        audio = mx.load(str(out / "audio_vae.safetensors"))
        voc = mx.load(str(out / "vocoder.safetensors"))
        assert "audio_vae.encoder.conv_in.conv.weight" in audio
        assert "audio_vae.decoder.conv_out.conv.weight" in audio
        assert "audio_vae.per_channel_statistics._mean_of_means" in audio
        assert "vocoder.resblocks.0.convs.0.weight" in voc
        assert "vocoder.ups.0.weight" in voc
        # 2D conv: (O, I, H, W) -> (O, H, W, I); 1D conv: (O, I, L) -> (O, L, I)
        assert audio["audio_vae.encoder.conv_in.conv.weight"].shape == (4, 3, 2)
        assert audio["audio_vae.decoder.conv_out.conv.weight"].shape == (2, 3, 3, 4)
        # Double ``vocoder.vocoder.`` prefix normalized to one; ConvTranspose1d
        # (I, O, L) -> (O, L, I); 3-D lowpass filters permuted like Conv1d.
        assert voc["vocoder.resblocks.0.convs.0.weight"].shape == (2, 3, 2)
        assert voc["vocoder.ups.0.weight"].shape == (4, 5, 6)
        assert voc["vocoder.act_post.upsample.filter"].shape == (1, 12, 1)
        # STFT basis matrices permuted like Conv1d: (F, 1, L) -> (F, L, 1).
        assert voc["vocoder.mel_stft.stft_fn.forward_basis"].shape == (4, 3, 1)

    def test_duration_head(self, tmp_path):
        from mlx.utils import tree_flatten

        src = tmp_path / "duration.safetensors"
        head = DurationHead()
        weights = {
            f"model.diffusion_model.duration_head.{k}": np.array(v, dtype=np.float32)
            for k, v in tree_flatten(head.parameters())
        }
        save_file(weights, str(src))
        out = tmp_path / "out"
        convert_duration_head(src, out)
        raw = mx.load(str(out / "duration_head.safetensors"))
        assert len(raw) == 15
        assert "duration_head.attention_pooler.cross_attn.in_proj_weight" in raw
        assert "duration_head.attention_pooler.query_tokens" in raw
        assert "duration_head.video_modality_emb" in raw

    def test_upscalers(self, tmp_path):
        def _mk(path, config):
            weights = {
                "initial_conv.weight": np.zeros((2, 2, 3), dtype=np.float32),
                "final_conv.weight": np.zeros((2, 2, 1, 1, 1), dtype=np.float32),
                "upsampler.0.weight": np.zeros((2, 2, 3, 3), dtype=np.float32),
                "final_conv.bias": np.zeros((2,), dtype=np.float32),
            }
            save_file(weights, str(path), metadata={"config": json.dumps(config)})

        spatial = tmp_path / "spatial.safetensors"
        temporal = tmp_path / "temporal.safetensors"
        _mk(spatial, {"in_channels": 2, "mid_channels": 4, "spatial_upsample": True})
        _mk(temporal, {"in_channels": 2, "mid_channels": 4, "temporal_upsample": True})
        out = tmp_path / "out"
        convert_upscalers(out, spatial, temporal)
        s = mx.load(str(out / "spatial_upscaler_x2.safetensors"))
        t = mx.load(str(out / "temporal_upscaler_x2.safetensors"))
        assert "initial_conv.weight" in s
        assert "final_conv.bias" in t
        # Conv3d (O, I, D, H, W) -> (O, D, H, W, I); Conv2d (O, I, H, W) -> (O, H, W, I).
        assert s["final_conv.weight"].shape == (2, 1, 1, 1, 2)
        assert s["upsampler.0.weight"].shape == (2, 3, 3, 2)
        # 3-D weights (e.g. blur kernels) are not conv weights: untouched.
        assert s["initial_conv.weight"].shape == (2, 2, 3)
        scfg = json.loads((out / "spatial_upscaler_x2_config.json").read_text())
        assert scfg["config"]["spatial_upsample"] is True
        assert (out / "temporal_upscaler_x2_config.json").exists()
