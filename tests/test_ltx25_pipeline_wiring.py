"""LTX-2.5 pipeline wiring: model-version detection + text encoder source selection."""

import json

from ltx_pipelines_mlx.utils._orchestration import detect_model_version
from ltx_pipelines_mlx.utils.blocks import PromptEncoder
from ltx_pipelines_mlx.utils.media_io import (
    DEFAULT_IMAGE_CRF,
    DEFAULT_IMAGE_CRF_V25,
    default_image_crf,
)

T25_TRANSFORMER = {
    "transformer": {
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "audio_cross_attention_dim": 2048,
        "use_keyframes_abs_pos_embedding": True,
    }
}


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class TestDetectModelVersion:
    def test_25_via_flag(self, tmp_path):
        _write(tmp_path / "embedded_config.json", T25_TRANSFORMER)
        assert detect_model_version(tmp_path) == "2.5.0"

    def test_25_via_model_version_field(self, tmp_path):
        _write(
            tmp_path / "config.json",
            {"model_version": "2.5.0", "num_layers": 48, "cross_attention_dim": 4096},
        )
        assert detect_model_version(tmp_path) == "2.5.0"

    def test_23_default(self, tmp_path):
        _write(
            tmp_path / "config.json",
            {"model_version": "2.3.0", "num_layers": 48, "cross_attention_dim": 4096},
        )
        assert detect_model_version(tmp_path) == "2.3"

    def test_23_no_flag(self, tmp_path):
        _write(tmp_path / "embedded_config.json", {"transformer": {"num_layers": 48}})
        assert detect_model_version(tmp_path) == "2.3"

    def test_no_config(self, tmp_path):
        assert detect_model_version(tmp_path) == "2.3"


class TestDefaultImageCrf:
    """Per-version I2V image CRF default (fix #4)."""

    def test_25_uses_18(self):
        assert DEFAULT_IMAGE_CRF_V25 == 18
        assert default_image_crf("2.5.0") == 18
        assert default_image_crf("2.5.1") == 18

    def test_23_keeps_33(self):
        assert default_image_crf("2.3") == DEFAULT_IMAGE_CRF == 33
        assert default_image_crf("") == 33

    def test_version_aware_resolution(self, tmp_path):
        """combined_image_conditionings picks 18 for a 2.5 model dir, 33 for 2.3."""
        import mlx.core as mx

        from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings

        (tmp_path / "embedded_config.json").write_text(
            json.dumps({"transformer": {"use_keyframes_abs_pos_embedding": True}})
        )

        class _Encoder:
            def encode(self, tensor):
                return mx.zeros((1, 128, 1, 1, 1))

        import ltx_pipelines_mlx.utils.media_io as media_io_mod

        orig = media_io_mod.load_image_and_preprocess
        seen = []

        def _fake_load(image_path, height, width, crf=DEFAULT_IMAGE_CRF):
            seen.append(crf)
            return mx.zeros((1, 3, height, width))

        media_io_mod.load_image_and_preprocess = _fake_load
        try:
            from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex
            from ltx_pipelines_mlx.utils.args import ImageConditioningInput

            img = ImageConditioningInput(path="x.png", frame_idx=0, strength=1.0)
            conds = combined_image_conditionings(
                [img],
                enc_h=32,
                enc_w=32,
                spatial_dims=(1, 1, 1),
                video_encoder=_Encoder(),
                frame_rate=24.0,
                model_dir=tmp_path,  # 2.5
            )
            assert seen and seen[0] == 18
            assert isinstance(conds[0], VideoConditionByLatentIndex)
            # 2.3 model dir → 33.
            (tmp_path / "embedded_config.json").write_text(json.dumps({"transformer": {}}))
            seen.clear()
            combined_image_conditionings(
                [img],
                enc_h=32,
                enc_w=32,
                spatial_dims=(1, 1, 1),
                video_encoder=_Encoder(),
                frame_rate=24.0,
                model_dir=tmp_path,
            )
            assert seen and seen[0] == 33
        finally:
            media_io_mod.load_image_and_preprocess = orig


class TestPromptEncoderTextEncoderSource:
    def test_25_local_gemma4_dir_wins(self, tmp_path):
        _write(
            tmp_path / "text_encoder" / "config.json",
            {"model_type": "gemma4", "text_config": {"num_hidden_layers": 48}},
        )
        encoder = PromptEncoder(model_dir=tmp_path)
        path, cls = encoder._text_encoder_source()
        assert path == str(tmp_path / "text_encoder")
        assert cls.__name__ == "Gemma4LanguageModel"

    def test_23_falls_back_to_gemma_model_id(self, tmp_path):
        encoder = PromptEncoder(model_dir=tmp_path, gemma_model_id="mlx-community/gemma-3-12b-it-4bit")
        path, cls = encoder._text_encoder_source()
        assert path == "mlx-community/gemma-3-12b-it-4bit"
        assert cls.__name__ == "GemmaLanguageModel"

    def test_local_dir_with_non_gemma4_config_falls_back(self, tmp_path):
        _write(tmp_path / "text_encoder" / "config.json", {"model_type": "gemma3"})
        encoder = PromptEncoder(model_dir=tmp_path)
        path, cls = encoder._text_encoder_source()
        assert path == encoder.gemma_model_id
        assert cls.__name__ == "GemmaLanguageModel"
