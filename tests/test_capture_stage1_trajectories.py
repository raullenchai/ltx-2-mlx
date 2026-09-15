from pathlib import Path

import pytest

from scripts.capture_stage1_trajectories import _reuse_condition, _reuse_manifest_prompts


def _write_condition(path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    save_file(
        {
            "video_prompt_embeds": np.zeros((1, 2), dtype=np.float32),
            "audio_prompt_embeds": np.zeros((1, 2), dtype=np.float32),
            "prompt_attention_mask": np.ones((1,), dtype=np.float32),
        },
        path,
    )


def test_reuses_stage2_condition_by_hard_link(tmp_path: Path) -> None:
    source = tmp_path / "source/.precomputed/conditions/condition_0007.safetensors"
    source.parent.mkdir(parents=True)
    _write_condition(source)

    destination = _reuse_condition(tmp_path / "source", tmp_path / "output", 7)

    assert source.stat().st_ino == destination.stat().st_ino
    assert _reuse_condition(tmp_path / "source", tmp_path / "output", 7) == destination


def test_rejects_different_existing_reused_condition(tmp_path: Path) -> None:
    source = tmp_path / "source/.precomputed/conditions/condition_0007.safetensors"
    source.parent.mkdir(parents=True)
    _write_condition(source)
    destination = tmp_path / "output/.precomputed/stage1_conditions/latent_0007.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"other")

    with pytest.raises(FileExistsError, match="different content"):
        _reuse_condition(tmp_path / "source", tmp_path / "output", 7)


def test_reads_prompt_identity_from_condition_source_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").write_text(
        '{"index":7,"prompt":"held-out prompt"}\n'
    )

    assert _reuse_manifest_prompts(tmp_path) == {7: "held-out prompt"}


def test_rejects_duplicate_condition_source_manifest_index(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").write_text(
        '{"index":7,"prompt":"first"}\n{"index":7,"prompt":"second"}\n'
    )

    with pytest.raises(ValueError, match="duplicate"):
        _reuse_manifest_prompts(tmp_path)
