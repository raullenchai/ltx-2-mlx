from pathlib import Path

import pytest

from scripts.capture_stage1_trajectories import _reuse_condition


def test_reuses_stage2_condition_by_hard_link(tmp_path: Path) -> None:
    source = tmp_path / "source/.precomputed/conditions/condition_0007.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"condition")

    destination = _reuse_condition(tmp_path / "source", tmp_path / "output", 7)

    assert destination.read_bytes() == b"condition"
    assert source.stat().st_ino == destination.stat().st_ino
    assert _reuse_condition(tmp_path / "source", tmp_path / "output", 7) == destination


def test_rejects_different_existing_reused_condition(tmp_path: Path) -> None:
    source = tmp_path / "source/.precomputed/conditions/condition_0007.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    destination = tmp_path / "output/.precomputed/stage1_conditions/latent_0007.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"other")

    with pytest.raises(FileExistsError, match="different content"):
        _reuse_condition(tmp_path / "source", tmp_path / "output", 7)
