"""Tests for trainer datasets and timestep samplers."""

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------
# PrecomputedDataset
# ---------------------------------------------------------------------------
class TestPrecomputedDataset:
    """Tests for PrecomputedDataset loading and normalization."""

    def _make_sample(self, directory, prefix, index, latent_shape=(128, 4, 16, 22), extra=None):
        """Helper: write a safetensors file with a latent array and optional extras."""
        data = {"latents": np.random.randn(*latent_shape).astype(np.float32)}
        if extra:
            data.update(extra)
        path = directory / f"{prefix}_{index:04d}.safetensors"
        save_file(data, str(path))
        return path

    def test_load_safetensors(self, tmp_path):
        """Create synthetic safetensors, verify loading returns correct shapes."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        latents_dir = tmp_path / "latents"
        conditions_dir = tmp_path / "conditions"
        latents_dir.mkdir()
        conditions_dir.mkdir()

        C, F, H, W = 128, 4, 16, 22

        seq_len, embed_dim = 256, 4096

        self._make_sample(latents_dir, "latent", 0, latent_shape=(C, F, H, W))
        save_file(
            {
                "video_prompt_embeds": np.random.randn(seq_len, embed_dim).astype(np.float32),
                "prompt_attention_mask": np.ones(seq_len, dtype=np.float32),
            },
            str(conditions_dir / "condition_0000.safetensors"),
        )

        ds = PrecomputedDataset(str(tmp_path))
        assert len(ds) == 1

        sample = ds[0]
        assert "latent_conditions" in sample
        assert "text_conditions" in sample

        latents = sample["latent_conditions"]["latents"]
        assert isinstance(latents, mx.array)
        assert latents.shape == (C, F, H, W)

        embeds = sample["text_conditions"]["video_prompt_embeds"]
        assert embeds.shape == (seq_len, embed_dim)

    def test_normalize_data_sources_none(self):
        """None defaults to latents -> latent_conditions, conditions -> text_conditions."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        result = PrecomputedDataset._normalize_data_sources(None)
        assert result == {"latents": "latent_conditions", "conditions": "text_conditions"}

    def test_normalize_data_sources_list(self):
        """List input maps each entry to itself."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        result = PrecomputedDataset._normalize_data_sources(["a", "b"])
        assert result == {"a": "a", "b": "b"}

    def test_normalize_data_sources_dict(self):
        """Dict input is returned as a copy."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        src = {"dir_a": "key_a"}
        result = PrecomputedDataset._normalize_data_sources(src)
        assert result == src
        assert result is not src  # must be a copy

    def test_legacy_patchified_format(self, tmp_path):
        """2D latents [seq_len, C] with metadata get converted to [C, F, H, W]."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        latents_dir = tmp_path / "latents"
        conditions_dir = tmp_path / "conditions"
        latents_dir.mkdir()
        conditions_dir.mkdir()

        F, H, W, C = 4, 8, 10, 128
        seq_len = F * H * W

        save_file(
            {
                "latents": np.random.randn(seq_len, C).astype(np.float32),
                "num_frames": np.array([F], dtype=np.int32),
                "height": np.array([H], dtype=np.int32),
                "width": np.array([W], dtype=np.int32),
            },
            str(latents_dir / "latent_0000.safetensors"),
        )
        save_file(
            {"video_prompt_embeds": np.random.randn(256, 4096).astype(np.float32)},
            str(conditions_dir / "condition_0000.safetensors"),
        )

        ds = PrecomputedDataset(str(tmp_path))
        sample = ds[0]
        latents = sample["latent_conditions"]["latents"]
        assert latents.shape == (C, F, H, W)

    def test_missing_directory_raises(self, tmp_path):
        """FileNotFoundError when data root does not exist."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        with pytest.raises(FileNotFoundError):
            PrecomputedDataset(str(tmp_path / "nonexistent"))

    def test_mismatched_sample_counts(self, tmp_path):
        """Different file counts across sources raises ValueError."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        latents_dir = tmp_path / "latents"
        conditions_dir = tmp_path / "conditions"
        latents_dir.mkdir()
        conditions_dir.mkdir()

        # Two latent files, one condition file
        self._make_sample(latents_dir, "sample", 0)
        self._make_sample(latents_dir, "sample", 1)
        save_file(
            {"video_prompt_embeds": np.zeros((10, 4096), dtype=np.float32)},
            str(conditions_dir / "sample_0000.safetensors"),
        )

        # The dataset discovers only matching pairs (sample_0000).
        # sample_0001 has no matching condition so it is skipped.
        ds = PrecomputedDataset(
            str(tmp_path),
            data_sources={"latents": "latent_conditions", "conditions": "text_conditions"},
        )
        assert len(ds) == 1

    def test_precomputed_subdir(self, tmp_path):
        """Data in .precomputed/ subdir is found when passing the parent path."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        precomp = tmp_path / ".precomputed"
        latents_dir = precomp / "latents"
        conditions_dir = precomp / "conditions"
        latents_dir.mkdir(parents=True)
        conditions_dir.mkdir(parents=True)

        self._make_sample(latents_dir, "latent", 0)
        save_file(
            {"video_prompt_embeds": np.zeros((10, 4096), dtype=np.float32)},
            str(conditions_dir / "condition_0000.safetensors"),
        )

        ds = PrecomputedDataset(str(tmp_path))
        assert len(ds) == 1

    def test_condition_latent_name_mapping(self, tmp_path):
        """latent_XXXX.safetensors maps to condition_XXXX.safetensors in conditions dir."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        latents_dir = tmp_path / "latents"
        conditions_dir = tmp_path / "conditions"
        latents_dir.mkdir()
        conditions_dir.mkdir()

        C, F, H, W = 128, 4, 8, 10
        self._make_sample(latents_dir, "latent", 5, latent_shape=(C, F, H, W))
        save_file(
            {"video_prompt_embeds": np.random.randn(256, 4096).astype(np.float32)},
            str(conditions_dir / "condition_0005.safetensors"),
        )

        ds = PrecomputedDataset(str(tmp_path))
        assert len(ds) == 1

        sample = ds[0]
        assert sample["latent_conditions"]["latents"].shape == (C, F, H, W)
        assert "video_prompt_embeds" in sample["text_conditions"]

    def test_sample_discovery_is_filename_deterministic(self, tmp_path):
        """Creation/glob order must not change sample indices across processes."""
        from ltx_trainer_mlx.datasets import PrecomputedDataset

        latents_dir = tmp_path / "latents"
        conditions_dir = tmp_path / "conditions"
        latents_dir.mkdir()
        conditions_dir.mkdir()
        for index in (9, 1, 5):
            self._make_sample(latents_dir, "latent", index)
            save_file(
                {"video_prompt_embeds": np.zeros((2, 4), dtype=np.float32)},
                str(conditions_dir / f"condition_{index:04d}.safetensors"),
            )

        dataset = PrecomputedDataset(str(tmp_path))
        assert [path.name for path in dataset.sample_files["latent_conditions"]] == [
            "latent_0001.safetensors",
            "latent_0005.safetensors",
            "latent_0009.safetensors",
        ]


# ---------------------------------------------------------------------------
# DummyDataset
# ---------------------------------------------------------------------------
class TestDummyDataset:
    """Tests for DummyDataset shape correctness and validation."""

    def test_shape_correctness(self):
        """Verify all output shapes match expected dimensions."""
        from ltx_trainer_mlx.datasets import DummyDataset

        ds = DummyDataset(
            width=512,
            height=256,
            num_frames=25,
            fps=24,
            dataset_length=5,
            latent_dim=128,
            latent_spatial_compression_ratio=32,
            latent_temporal_compression_ratio=8,
            prompt_embed_dim=4096,
            audio_embed_dim=2048,
            prompt_sequence_length=256,
        )
        sample = ds[0]

        # Latent: (C, F_latent, H_latent, W_latent)
        latents = sample["latent_conditions"]["latents"]
        expected_F = (25 - 1) // 8 + 1  # 4
        expected_H = 256 // 32  # 8
        expected_W = 512 // 32  # 16
        assert latents.shape == (128, expected_F, expected_H, expected_W)

        # Text embeddings
        video_embeds = sample["text_conditions"]["video_prompt_embeds"]
        assert video_embeds.shape == (256, 4096)

        audio_embeds = sample["text_conditions"]["audio_prompt_embeds"]
        assert audio_embeds.shape == (256, 2048)

        mask = sample["text_conditions"]["prompt_attention_mask"]
        assert mask.shape == (256,)
        assert mask.dtype == mx.bool_

    def test_length(self):
        """len() returns configured dataset_length."""
        from ltx_trainer_mlx.datasets import DummyDataset

        ds = DummyDataset(dataset_length=42)
        assert len(ds) == 42

    def test_invalid_width(self):
        """Width not divisible by 32 raises ValueError."""
        from ltx_trainer_mlx.datasets import DummyDataset

        with pytest.raises(ValueError, match="Width must be divisible by 32"):
            DummyDataset(width=500)

    def test_invalid_height(self):
        """Height not divisible by 32 raises ValueError."""
        from ltx_trainer_mlx.datasets import DummyDataset

        with pytest.raises(ValueError, match="Height must be divisible by 32"):
            DummyDataset(height=500)

    def test_invalid_frames(self):
        """Frames not satisfying % 8 == 1 raises ValueError."""
        from ltx_trainer_mlx.datasets import DummyDataset

        with pytest.raises(ValueError, match="remainder of 1"):
            DummyDataset(num_frames=24)


# ---------------------------------------------------------------------------
# Timestep Samplers
# ---------------------------------------------------------------------------
class TestTimestepSamplers:
    """Tests for uniform and shifted logit-normal timestep samplers."""

    def test_uniform_range(self):
        """All sampled values in [min, max]."""
        from ltx_trainer_mlx.timestep_samplers import UniformTimestepSampler

        sampler = UniformTimestepSampler(min_value=0.2, max_value=0.8)
        samples = sampler.sample(1000)
        mx.eval(samples)
        assert float(mx.min(samples)) >= 0.2
        assert float(mx.max(samples)) <= 0.8

    def test_uniform_shape(self):
        """Output shape matches batch_size."""
        from ltx_trainer_mlx.timestep_samplers import UniformTimestepSampler

        sampler = UniformTimestepSampler()
        samples = sampler.sample(16)
        assert samples.shape == (16,)

    def test_uniform_sample_for(self):
        """sample_for with a batch tensor returns correct shape."""
        from ltx_trainer_mlx.timestep_samplers import UniformTimestepSampler

        sampler = UniformTimestepSampler()
        batch = mx.zeros((8, 100, 128))
        samples = sampler.sample_for(batch)
        assert samples.shape == (8,)

    def test_uniform_sample_for_wrong_ndim(self):
        """sample_for raises ValueError for non-3D input."""
        from ltx_trainer_mlx.timestep_samplers import UniformTimestepSampler

        sampler = UniformTimestepSampler()
        with pytest.raises(ValueError, match="3 dimensions"):
            sampler.sample_for(mx.zeros((8, 100)))

    def test_shifted_logit_normal_range(self):
        """All values in [0, 1]."""
        from ltx_trainer_mlx.timestep_samplers import ShiftedLogitNormalTimestepSampler

        sampler = ShiftedLogitNormalTimestepSampler()
        samples = sampler.sample(1000, seq_length=2048)
        mx.eval(samples)
        assert float(mx.min(samples)) >= 0.0
        assert float(mx.max(samples)) <= 1.0

    def test_shifted_logit_normal_shape(self):
        """Output shape matches batch_size."""
        from ltx_trainer_mlx.timestep_samplers import ShiftedLogitNormalTimestepSampler

        sampler = ShiftedLogitNormalTimestepSampler()
        samples = sampler.sample(32, seq_length=2048)
        assert samples.shape == (32,)

    def test_shifted_logit_normal_requires_seq_length(self):
        """Raises ValueError when seq_length is None."""
        from ltx_trainer_mlx.timestep_samplers import ShiftedLogitNormalTimestepSampler

        sampler = ShiftedLogitNormalTimestepSampler()
        with pytest.raises(ValueError, match="seq_length is required"):
            sampler.sample(8)

    def test_shifted_logit_normal_sample_for(self):
        """sample_for with a 3D batch tensor returns correct shape."""
        from ltx_trainer_mlx.timestep_samplers import ShiftedLogitNormalTimestepSampler

        sampler = ShiftedLogitNormalTimestepSampler()
        batch = mx.zeros((4, 2048, 128))
        samples = sampler.sample_for(batch)
        assert samples.shape == (4,)

    def test_sampler_registry(self):
        """SAMPLERS dict contains expected entries."""
        from ltx_trainer_mlx.timestep_samplers import (
            SAMPLERS,
            ShiftedLogitNormalTimestepSampler,
            UniformTimestepSampler,
        )

        assert "uniform" in SAMPLERS
        assert "shifted_logit_normal" in SAMPLERS
        assert SAMPLERS["uniform"] is UniformTimestepSampler
        assert SAMPLERS["shifted_logit_normal"] is ShiftedLogitNormalTimestepSampler

    def test_shift_interpolation(self):
        """Verify shift calculation at boundary values."""
        from ltx_trainer_mlx.timestep_samplers import ShiftedLogitNormalTimestepSampler

        shift_fn = ShiftedLogitNormalTimestepSampler._get_shift_for_sequence_length
        assert shift_fn(1024) == pytest.approx(0.95)
        assert shift_fn(4096) == pytest.approx(2.05)
        # Midpoint
        mid = shift_fn(2560)
        assert mid == pytest.approx(1.5, abs=0.01)
