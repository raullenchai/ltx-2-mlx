"""DurationHead (LTX-2.5) — module shapes, weight keys, grid snapping."""

import mlx.core as mx
import pytest

from ltx_core_mlx.model.duration_head import (
    AttentionPooler,
    DurationHead,
    load_duration_head,
    seconds_to_num_frames,
)


class TestDurationHeadModule:
    def test_weight_keys_match_spec(self):
        """15 tensors with the official duration_head.* names.

        Note: the spec says 16 but the official header carries 15 tensors
        (no separate cross_attn q/k/v biases — fused ``in_proj_weight`` /
        ``in_proj_bias`` only); the module matches the header 1:1."""
        from mlx.utils import tree_flatten

        head = DurationHead()
        keys = dict(tree_flatten(head.parameters()))
        assert len(keys) == 15
        expected_prefixes = {
            "video_input_proj.",
            "audio_input_proj.",
            "video_modality_emb",
            "audio_modality_emb",
            "attention_pooler.query_tokens",
            "attention_pooler.cross_attn.in_proj_weight",
            "attention_pooler.cross_attn.in_proj_bias",
            "attention_pooler.cross_attn.out_proj.weight",
            "attention_pooler.cross_attn.out_proj.bias",
            "mlp_hidden.",
            "mlp_out.",
        }
        for prefix in expected_prefixes:
            assert any(k.startswith(prefix) for k in keys), prefix

    def test_spec_shapes(self):
        from mlx.utils import tree_flatten

        head = DurationHead()
        keys = dict(tree_flatten(head.parameters()))
        assert keys["video_input_proj.weight"].shape == (256, 4096)
        assert keys["audio_input_proj.weight"].shape == (256, 2048)
        assert keys["video_modality_emb"].shape == (256,)
        assert keys["audio_modality_emb"].shape == (256,)
        assert keys["attention_pooler.query_tokens"].shape == (1, 256)
        assert keys["attention_pooler.cross_attn.in_proj_weight"].shape == (768, 256)
        assert keys["attention_pooler.cross_attn.in_proj_bias"].shape == (768,)
        assert keys["attention_pooler.cross_attn.out_proj.weight"].shape == (256, 256)
        assert keys["mlp_hidden.weight"].shape == (256, 256)
        assert keys["mlp_out.weight"].shape == (1, 256)

    def test_forward_video_only(self):
        head = DurationHead()
        video = mx.random.normal((2, 64, 4096))
        seconds = head(video_tokens=video)
        mx.synchronize()
        assert seconds.shape == (2,)
        assert (seconds > 0).all().item()  # exp() output

    def test_forward_audio_only(self):
        head = DurationHead()
        audio = mx.random.normal((1, 32, 2048))
        seconds = head(audio_tokens=audio)
        mx.synchronize()
        assert seconds.shape == (1,)

    def test_forward_both(self):
        head = DurationHead()
        video = mx.random.normal((1, 64, 4096))
        audio = mx.random.normal((1, 64, 2048))
        seconds = head(video_tokens=video, audio_tokens=audio)
        mx.synchronize()
        assert seconds.shape == (1,)

    def test_forward_requires_input(self):
        head = DurationHead()
        with pytest.raises(ValueError, match="at least one"):
            head()

    def test_pooler_cross_attention(self):
        pooler = AttentionPooler(hidden_dim=32, num_queries=1, num_heads=4)
        out = pooler(mx.random.normal((2, 16, 32)))
        mx.synchronize()
        assert out.shape == (2, 1, 32)


class TestSecondsToNumFrames:
    def test_grid_8k_plus_1(self):
        # 121 frames @24fps = 5.04s -> snap to 121 (already on grid)
        assert seconds_to_num_frames(5.04, 24, 1.0, 10.0) == 121
        # 5.0s -> 120 frames -> floor to 8k+1 grid: 113
        assert seconds_to_num_frames(5.0, 24, 1.0, 10.0) == 113
        # 1.0s -> 24 frames -> 17 on the grid, but < min_frames -> bump to 25
        assert seconds_to_num_frames(1.0, 24, 1.0, 10.0) == 25

    def test_clamps(self):
        # Below min -> bumped up to next grid point >= min_frames
        assert seconds_to_num_frames(0.1, 24, 1.0, 10.0) == 25
        # Above max -> clamped to 240 frames then floored to the grid: 233
        assert seconds_to_num_frames(100.0, 24, 1.0, 10.0) == 233

    def test_returns_odd_frames(self):
        for s in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
            frames = seconds_to_num_frames(s, 24, 1.0, 10.0)
            assert (frames - 1) % 8 == 0, f"{s}s -> {frames}"


class TestLoadDurationHead:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_duration_head(tmp_path) is None

    def test_loads_official_prefix(self, tmp_path):
        """duration_head.* keys (official single-file layout) load correctly."""
        import numpy as np
        from mlx.utils import tree_flatten
        from safetensors.numpy import save_file

        head = DurationHead()
        weights = {
            f"duration_head.{k}": np.array(v, dtype=np.float32)
            for k, v in tree_flatten(head.parameters())
        }
        save_file(weights, str(tmp_path / "duration_head.safetensors"))

        loaded = load_duration_head(tmp_path)
        assert loaded is not None
        # A forward through the loaded head must run.
        video = mx.random.normal((1, 16, 4096))
        seconds = loaded(video_tokens=video)
        mx.synchronize()
        assert seconds.shape == (1,)

    def test_loads_diffusers_prefix(self, tmp_path):
        """model.diffusion_model.duration_head.* keys also load."""
        import numpy as np
        from mlx.utils import tree_flatten
        from safetensors.numpy import save_file

        head = DurationHead()
        weights = {
            f"model.diffusion_model.duration_head.{k}": np.array(v, dtype=np.float32)
            for k, v in tree_flatten(head.parameters())
        }
        save_file(weights, str(tmp_path / "duration_head.safetensors"))
        loaded = load_duration_head(tmp_path)
        assert loaded is not None
