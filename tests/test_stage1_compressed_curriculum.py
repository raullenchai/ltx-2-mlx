import os
import subprocess
import sys
from argparse import ArgumentTypeError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from ltx_core_mlx.loader.fast_stage1_segmented import FastStage1Segment, FastStage1SegmentedPackage
from scripts.render_segmented_stage1_ablation import (
    apply_diagnostic_base_schedule,
    apply_diagnostic_base_spans,
    parse_base_schedule,
)
from scripts.train_stage1_compressed_curriculum import (
    PHASES,
    PRIMARY_PHASES,
    REPLAY_PHASES,
    build_config,
    require_phase_checkpoint,
    validate_phase_checkpoint,
)
from scripts.train_stage1_segmented_curriculum import select_segment_phases


def test_evaluator_supports_direct_script_entrypoint(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts/evaluate_stage1_compressed_curriculum.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(repository / path)
        for path in (
            "packages/ltx-core-mlx/src",
            "packages/ltx-pipelines-mlx/src",
            "packages/ltx-trainer/src",
        )
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Evaluate one shared stage-1 adapter" in result.stdout


def test_segmented_evaluator_supports_direct_script_entrypoint(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts/evaluate_stage1_segmented_curriculum.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(repository / path)
        for path in (
            "packages/ltx-core-mlx/src",
            "packages/ltx-pipelines-mlx/src",
            "packages/ltx-trainer/src",
        )
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Evaluate three independent Stage-1 adapters" in result.stdout


def test_segmented_ablation_renderer_supports_direct_script_entrypoint(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts/render_segmented_stage1_ablation.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(repository / path)
        for path in (
            "packages/ltx-core-mlx/src",
            "packages/ltx-pipelines-mlx/src",
            "packages/ltx-trainer/src",
        )
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--base-span {0-3,3-5,5-7}" in result.stdout


def test_segmented_ablation_requires_diagnostic_package() -> None:
    segments = (SimpleNamespace(start_index=0, end_index=3),)
    pipe = SimpleNamespace(
        _fast_stage1_segmented_package=SimpleNamespace(
            segments=segments,
            qualification_revision="diagnostic-control",
        )
    )

    apply_diagnostic_base_spans(pipe, ((0, 3),))
    assert pipe._diagnostic_base_stage1_spans == frozenset({(0, 3)})

    pipe._fast_stage1_segmented_package.qualification_revision = "release-v1"
    with pytest.raises(ValueError, match="diagnostic qualification"):
        apply_diagnostic_base_spans(pipe, ((0, 3),))
    with pytest.raises(ValueError, match="packaged Stage-1 segments"):
        apply_diagnostic_base_spans(
            SimpleNamespace(
                _fast_stage1_segmented_package=SimpleNamespace(
                    segments=segments,
                    qualification_revision="diagnostic-control",
                )
            ),
            ((3, 5),),
        )


def test_segmented_ablation_supports_custom_all_base_schedule() -> None:
    segment = FastStage1Segment(
        adapter_path=Path("/model/inert.safetensors"),
        artifact_sha256="0" * 64,
        start_index=0,
        end_index=3,
        sigma=1.0,
        target_sigma=0.98125,
        lora_rank=8,
        lora_alpha=8.0,
    )
    package = FastStage1SegmentedPackage(
        segments=(segment,),
        transformer_path=Path("/model/transformer.safetensors"),
        schedule=(1.0, 0.98125, 0.909375, 0.421875, 0.0),
        noise_reference_sigmas=(1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
        noise_total_steps=8,
        base_model_id="model",
        base_revision="1" * 40,
        transformer_sha256="2" * 64,
        transformer_config_sha256="3" * 64,
        pipeline_family="distilled_two_stage_ltx25",
        runtime_contract_major=1,
        qualification_revision="diagnostic-control",
    )
    pipe = SimpleNamespace(_fast_stage1_segmented_package=package)

    boundaries = parse_base_schedule("0,1,5,7,8")
    apply_diagnostic_base_schedule(pipe, boundaries)

    configured = pipe._fast_stage1_segmented_package
    assert configured.schedule == (1.0, 0.99375, 0.909375, 0.421875, 0.0)
    assert [(item.start_index, item.end_index) for item in configured.segments] == [(0, 1), (1, 5), (5, 7)]
    assert pipe._diagnostic_base_stage1_spans == frozenset({(0, 1), (1, 5), (5, 7)})
    with pytest.raises(ArgumentTypeError, match="0 through 8"):
        parse_base_schedule("1,5,8")


def test_segmented_trainer_exposes_single_span_control(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts/train_stage1_segmented_curriculum.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(repository / path)
        for path in (
            "packages/ltx-core-mlx/src",
            "packages/ltx-pipelines-mlx/src",
            "packages/ltx-trainer/src",
        )
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--span {0-3,3-5,5-7,1-3}" in result.stdout
    assert "--steps STEPS" in result.stdout


def test_segmented_trainer_can_scale_one_span_budget() -> None:
    phases = select_segment_phases(["0-3"], 600)

    assert len(phases) == 1
    assert phases[0].name == "primary-0-3"
    assert phases[0].steps == 600
    assert phases[0].checkpoint_interval == 600

    diagnostic = select_segment_phases(["1-3"], 100)
    assert [(phase.start_index, phase.target_index) for phase in diagnostic] == [(1, 3)]
    assert diagnostic[0].name == "diagnostic-1-3"

    with pytest.raises(ValueError, match="exactly one"):
        select_segment_phases(None, 600)
    with pytest.raises(ValueError, match="exactly one"):
        select_segment_phases(["0-3", "3-5"], 600)
    with pytest.raises(ValueError, match="positive"):
        select_segment_phases(["0-3"], 0)


def test_curriculum_uses_selected_boundaries_and_original_noise_lanes() -> None:
    assert [(p.start_index, p.target_index) for p in PRIMARY_PHASES] == [(0, 3), (3, 5), (5, 7), (7, 8)]
    assert [p.noise_step_index for p in PRIMARY_PHASES] == [0, 3, 5, None]
    assert [(p.sigma, p.target_sigma) for p in PRIMARY_PHASES] == [
        (1.0, 0.98125),
        (0.98125, 0.909375),
        (0.909375, 0.421875),
        (0.421875, 0.0),
    ]
    assert len(PHASES) == 8


def test_ancestral_phase_config_is_noise_coupled_and_resumes() -> None:
    checkpoint = Path("/tmp/previous.safetensors")
    phase = PRIMARY_PHASES[1]
    config = build_config(
        phase=phase,
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/stage1"),
        output=Path("/output/phase"),
        load_checkpoint=checkpoint,
    )

    assert config["model"]["load_checkpoint"] == str(checkpoint)
    assert config["training_strategy"] == {
        "name": "stage1_transition_distill",
        "sigma": 0.98125,
        "target_sigma": 0.909375,
        "video_start_latents_dir": "stage1_video_step_03",
        "video_terminal_latents_dir": "stage1_video_step_05",
        "audio_start_latents_dir": "stage1_audio_step_03",
        "audio_terminal_latents_dir": "stage1_audio_step_05",
        "conditions_dir": "stage1_conditions",
        "video_loss_weight": 1.0,
        "audio_loss_weight": 1.0,
        "curriculum_adapter_mode": "shared",
        "ancestral_noise_step_index": 3,
        "ancestral_noise_total_steps": 8,
        "ancestral_eta": 1.0,
        "ancestral_s_noise": 1.0,
    }


def test_span_v2_phase_config_couples_every_fine_noise_lane() -> None:
    phase = PRIMARY_PHASES[0]
    config = build_config(
        phase=phase,
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/stage1"),
        output=Path("/output/phase"),
        load_checkpoint=None,
        noise_coupling="span-v2",
    )

    strategy = config["training_strategy"]
    assert strategy["ancestral_noise_step_index"] == 0
    assert strategy["ancestral_noise_step_end_index"] == 3
    assert strategy["ancestral_noise_reference_sigmas"] == [
        1.0,
        0.99375,
        0.9875,
        0.98125,
        0.975,
        0.909375,
        0.725,
        0.421875,
        0.0,
    ]


def test_independent_segment_config_starts_from_clean_base() -> None:
    config = build_config(
        phase=PRIMARY_PHASES[1],
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/stage1"),
        output=Path("/output/phase"),
        load_checkpoint=None,
        noise_coupling="span-v2",
        adapter_mode="independent",
    )

    assert "load_checkpoint" not in config["model"]
    assert config["training_strategy"]["curriculum_adapter_mode"] == "independent"


def test_terminal_phase_draws_no_noise_and_replay_is_low_lr() -> None:
    config = build_config(
        phase=PRIMARY_PHASES[-1],
        model=Path("/models/ltx"),
        transformer_file="transformer.safetensors",
        data=Path("/data/stage1"),
        output=Path("/output/terminal"),
        load_checkpoint=None,
    )
    assert config["training_strategy"]["target_sigma"] == 0.0
    assert "ancestral_noise_step_index" not in config["training_strategy"]
    assert all(phase.learning_rate == 1.0e-6 and phase.steps == 24 for phase in REPLAY_PHASES)


def test_phase_checkpoint_validation_rejects_stale_transition(tmp_path) -> None:
    phase = PRIMARY_PHASES[0]
    metadata = {
        "distillation": "stage1_transition",
        "stage1_sigma": "1.0",
        "stage1_target_sigma": "0.98125",
        "stage1_video_start_latents_dir": "stage1_video_step_00",
        "stage1_video_target_latents_dir": "stage1_video_step_03",
        "stage1_audio_start_latents_dir": "stage1_audio_step_00",
        "stage1_audio_target_latents_dir": "stage1_audio_step_03",
        "stage1_sampler": "ancestral",
        "stage1_noise_step_index": "0",
        "stage1_noise_total_steps": "8",
        "lora_rank": "8",
        "lora_alpha": "8",
    }
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file({"weight": np.zeros((1,), dtype=np.float32)}, checkpoint, metadata=metadata)
    validate_phase_checkpoint(checkpoint, phase)

    metadata["stage1_target_sigma"] = "0.909375"
    save_file({"weight": np.zeros((1,), dtype=np.float32)}, checkpoint, metadata=metadata)
    with pytest.raises(ValueError, match="stage1_target_sigma"):
        validate_phase_checkpoint(checkpoint, phase)


def test_fresh_span_checkpoint_is_validated_with_selected_coupling(tmp_path) -> None:
    phase = PRIMARY_PHASES[0]
    metadata = {
        "distillation": "stage1_transition",
        "stage1_sigma": "1.0",
        "stage1_target_sigma": "0.98125",
        "stage1_video_start_latents_dir": "stage1_video_step_00",
        "stage1_video_target_latents_dir": "stage1_video_step_03",
        "stage1_audio_start_latents_dir": "stage1_audio_step_00",
        "stage1_audio_target_latents_dir": "stage1_audio_step_03",
        "stage1_sampler": "ancestral_span_v2",
        "stage1_noise_step_index": "0",
        "stage1_noise_step_end_index": "3",
        "stage1_noise_total_steps": "8",
        "stage1_noise_reference_sigmas": "[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]",
        "stage1_curriculum_noise_coupling": "span-v2",
        "lora_rank": "8",
        "lora_alpha": "8",
    }
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file({"weight": np.zeros((1,), dtype=np.float32)}, checkpoint, metadata=metadata)

    assert require_phase_checkpoint(checkpoint, phase, "span-v2") == checkpoint
    with pytest.raises(ValueError, match="ancestral noise coupling"):
        require_phase_checkpoint(checkpoint, phase, "lane")


def test_independent_resume_rejects_shared_checkpoint(tmp_path) -> None:
    phase = PRIMARY_PHASES[0]
    metadata = {
        "distillation": "stage1_transition",
        "stage1_sigma": "1.0",
        "stage1_target_sigma": "0.98125",
        "stage1_video_start_latents_dir": "stage1_video_step_00",
        "stage1_video_target_latents_dir": "stage1_video_step_03",
        "stage1_audio_start_latents_dir": "stage1_audio_step_00",
        "stage1_audio_target_latents_dir": "stage1_audio_step_03",
        "stage1_sampler": "ancestral_span_v2",
        "stage1_noise_step_index": "0",
        "stage1_noise_step_end_index": "3",
        "stage1_noise_total_steps": "8",
        "stage1_noise_reference_sigmas": "[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]",
        "stage1_curriculum_noise_coupling": "span-v2",
        "stage1_curriculum_adapter_mode": "shared",
        "lora_rank": "8",
        "lora_alpha": "8",
    }
    checkpoint = tmp_path / "checkpoint.safetensors"
    save_file({"weight": np.zeros((1,), dtype=np.float32)}, checkpoint, metadata=metadata)

    with pytest.raises(ValueError, match="independent adapter mode"):
        require_phase_checkpoint(checkpoint, phase, "span-v2", "independent")
