from __future__ import annotations

import json

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from scripts.package_fast_stage2 import main


def test_packages_terminal_stage2_checkpoint(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model"
    output = tmp_path / "package"
    model.mkdir()
    (model / "embedded_config.json").write_text("{}")
    (model / "transformer-distilled.safetensors").write_bytes(b"transformer")
    checkpoint = tmp_path / "student.safetensors"
    save_file(
        {"block.to_q.lora_A.weight": np.zeros((2, 4), dtype=np.float32)},
        checkpoint,
        metadata={
            "stage2_sigma": "0.909375",
            "stage2_target_sigma": "0.0",
            "lora_rank": "2",
            "lora_alpha": "2",
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "package_fast_stage2.py",
            "--model-dir",
            str(model),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--base-model-id",
            "example/ltx",
            "--base-revision",
            "a" * 40,
            "--qualification-revision",
            "terminal-qual-v1",
        ],
    )

    assert main() == 0

    with safe_open(output / "fast-stage2.safetensors", framework="numpy") as packaged:
        metadata = packaged.metadata()
    assert metadata is not None
    assert metadata["fast_stage2_capability"] == "ltx_stage2_terminal_v1"
    assert json.loads(metadata["fast_stage2_schedule"]) == [0.909375, 0.0]
    assert metadata["stage2_steps"] == "1"
    assert json.loads((output / "fast-stage2.json").read_text())["schema_version"] == 1
