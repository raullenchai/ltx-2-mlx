import json
from pathlib import Path

import pytest

from scripts.render_combined_fast_blind_suite import (
    CasePlan,
    _case_plans,
    _completed_timing,
    _environment,
    _generation_command,
    _review_case,
)


def test_review_case_does_not_reveal_mapping_or_timings(tmp_path: Path) -> None:
    plan = CasePlan(3, "A detailed moving market", 45, fast_is_a=True, fast_first=False)

    case = _review_case(plan, tmp_path / "blind")
    rendered = json.dumps(case)

    assert case["A"] == "case-03/A.mp4"
    assert case["seed"] == 45
    assert "fast" not in rendered
    assert "standard" not in rendered
    assert "seconds" not in rendered


def test_case_plans_are_deterministic_and_use_distinct_seeds() -> None:
    first = _case_plans(["one", "two"], seed=100, mapping_seed=7)
    second = _case_plans(["one", "two"], seed=100, mapping_seed=7)

    assert first == second
    assert [plan.seed for plan in first] == [100, 101]


def test_generation_command_only_adds_fast_contract_for_fast_variant(tmp_path: Path) -> None:
    plan = CasePlan(0, "moving subject", 42, fast_is_a=False, fast_first=True)
    common = dict(
        model="model",
        output=tmp_path / "out.mp4",
        plan=plan,
        width=768,
        height=512,
        frames=241,
        frame_rate=24,
    )

    standard = _generation_command(**common)
    fast = _generation_command(
        **common,
        fast_stage1_manifest="fast-stage1.json",
        fast_stage2_manifest="fast-stage2.json",
    )

    assert "--distilled" in standard
    assert "--low-ram" in standard
    assert "--fast-stage1-manifest" not in standard
    assert fast[-4:] == [
        "--fast-stage1-manifest",
        "fast-stage1.json",
        "--fast-stage2-manifest",
        "fast-stage2.json",
    ]

    segmented = _generation_command(
        **common,
        fast_stage1_segmented_manifest="fast-stage1-segmented.json",
        fast_stage2_manifest="fast-stage2.json",
    )
    assert segmented[-4:] == [
        "--fast-stage1-segmented-manifest",
        "fast-stage1-segmented.json",
        "--fast-stage2-manifest",
        "fast-stage2.json",
    ]


def test_generation_command_rejects_two_stage1_manifests(tmp_path: Path) -> None:
    plan = CasePlan(0, "prompt", 42, fast_is_a=False, fast_first=False)
    with pytest.raises(ValueError, match="either"):
        _generation_command(
            model="model",
            output=tmp_path / "out.mp4",
            plan=plan,
            width=768,
            height=512,
            frames=241,
            frame_rate=24,
            fast_stage1_manifest="fast-stage1.json",
            fast_stage1_segmented_manifest="fast-stage1-segmented.json",
            fast_stage2_manifest="fast-stage2.json",
        )


def test_generation_command_rejects_partial_fast_configuration(tmp_path: Path) -> None:
    plan = CasePlan(0, "prompt", 42, fast_is_a=False, fast_first=False)
    with pytest.raises(ValueError, match="both manifests"):
        _generation_command(
            model="model",
            output=tmp_path / "out.mp4",
            plan=plan,
            width=768,
            height=512,
            frames=241,
            frame_rate=24,
            fast_stage1_manifest="fast-stage1.json",
        )


def test_timing_checkpoint_is_fail_closed(tmp_path: Path) -> None:
    timing = tmp_path / ".timing.json"
    timing.write_text('{"standard_seconds": 10, "fast_seconds": 5}')
    assert _completed_timing(timing) == {"standard_seconds": 10.0, "fast_seconds": 5.0}

    timing.write_text('{"standard_seconds": 10}')
    with pytest.raises(ValueError, match="invalid timing checkpoint"):
        _completed_timing(timing)


def test_environment_is_diagnostic_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.render_combined_fast_blind_suite._sysctl",
        lambda name: {"hw.memsize": "51539607552", "machdep.cpu.brand_string": "Apple M4 Pro"}[name],
    )

    environment = _environment()

    assert environment["chip"] == "Apple M4 Pro"
    assert environment["memory_bytes"] == 51539607552
