from pathlib import Path

import pytest

from scripts.materialize_stage1_student_rollout import (
    _hardlink_dataset_view,
    _sample_index,
    _target_step,
)


def test_hardlink_view_excludes_only_replaced_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source/.precomputed"
    files = {
        "stage1_video_step_01/latent_0000.safetensors": b"start-video",
        "stage1_audio_step_01/latent_0000.safetensors": b"start-audio",
        "stage1_video_step_03/latent_0000.safetensors": b"teacher-video",
        "stage1_audio_step_03/latent_0000.safetensors": b"teacher-audio",
        "stage1_video_step_05/latent_0000.safetensors": b"later-teacher-video",
        "stage1_conditions/latent_0000.safetensors": b"conditions",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "source/manifest.jsonl").write_text('{"index":0}\n')
    (tmp_path / "source/student-rollout.json").write_text("old report\n")
    (tmp_path / "source/teacher-correction.json").write_text("old correction\n")

    linked = _hardlink_dataset_view(
        tmp_path / "source",
        tmp_path / "output",
        {"stage1_video_step_03", "stage1_audio_step_03"},
    )

    assert linked == 5
    assert not (tmp_path / "output/.precomputed/stage1_video_step_03").exists()
    assert not (tmp_path / "output/.precomputed/stage1_audio_step_03").exists()
    later = tmp_path / "output/.precomputed/stage1_video_step_05/latent_0000.safetensors"
    assert later.stat().st_ino == (source / "stage1_video_step_05/latent_0000.safetensors").stat().st_ino
    assert (tmp_path / "output/manifest.jsonl").stat().st_ino == (tmp_path / "source/manifest.jsonl").stat().st_ino
    assert not (tmp_path / "output/student-rollout.json").exists()
    assert not (tmp_path / "output/teacher-correction.json").exists()


def test_hardlink_view_rejects_existing_output(tmp_path: Path) -> None:
    (tmp_path / "source/.precomputed").mkdir(parents=True)
    (tmp_path / "output").mkdir()

    with pytest.raises(FileExistsError, match="existing rollout"):
        _hardlink_dataset_view(tmp_path / "source", tmp_path / "output", set())


def test_hardlink_view_rejects_output_inside_source(tmp_path: Path) -> None:
    (tmp_path / "source/.precomputed").mkdir(parents=True)

    with pytest.raises(ValueError, match="inside the source"):
        _hardlink_dataset_view(tmp_path / "source", tmp_path / "source/output", set())


def test_target_step_requires_matching_captured_sources() -> None:
    metadata = {
        "stage1_video_start_latents_dir": "stage1_video_step_01",
        "stage1_audio_start_latents_dir": "stage1_audio_step_01",
        "stage1_video_target_latents_dir": "stage1_video_step_03",
        "stage1_audio_target_latents_dir": "stage1_audio_step_03",
        "stage1_noise_step_index": "1",
        "stage1_noise_step_end_index": "3",
    }
    assert _target_step(metadata) == 3

    metadata["stage1_audio_target_latents_dir"] = "stage1_audio_step_04"
    with pytest.raises(ValueError, match="do not match"):
        _target_step(metadata)

    metadata["stage1_audio_target_latents_dir"] = "stage1_audio_step_03"
    metadata["stage1_noise_step_index"] = "0"
    with pytest.raises(ValueError, match="start source"):
        _target_step(metadata)


@pytest.mark.parametrize("name,index", [("latent_0000.safetensors", 0), ("latent_0123.safetensors", 123)])
def test_sample_index_uses_captured_filename(name: str, index: int) -> None:
    assert _sample_index(Path(name)) == index


def test_sample_index_rejects_nested_or_ambiguous_names() -> None:
    with pytest.raises(ValueError, match="nested"):
        _sample_index(Path("nested/latent_0001.safetensors"))
    with pytest.raises(ValueError, match="invalid"):
        _sample_index(Path("sample.safetensors"))
