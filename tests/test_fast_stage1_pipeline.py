from pathlib import Path
from types import MethodType, SimpleNamespace

from ltx_pipelines_mlx.distilled import DistilledPipeline


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
