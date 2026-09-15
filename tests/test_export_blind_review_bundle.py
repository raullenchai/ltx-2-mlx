from __future__ import annotations

import json

import pytest

from scripts.export_blind_review_bundle import export_bundle


def _source(tmp_path):
    source = tmp_path / "source"
    case = source / "case-00"
    case.mkdir(parents=True)
    for name in ("A.mp4", "B.mp4", "AB-side-by-side-muted.mp4"):
        (case / name).write_bytes(name.encode())
    (case / "sample-00-teacher.mp4").write_bytes(b"secret")
    (source / ".blind-mapping.json").write_text('{"0":{"A":"teacher"}}')
    (source / "anonymous-metrics.json").write_text('{"mapping_read":false}')
    review = {
        "blinded": True,
        "cases": [
            {
                "index": 0,
                "A": "case-00/A.mp4",
                "B": "case-00/B.mp4",
                "side_by_side_muted": "case-00/AB-side-by-side-muted.mp4",
            }
        ],
    }
    (source / "review-index.json").write_text(json.dumps(review))
    return source


def test_exports_only_anonymous_review_surface(tmp_path) -> None:
    output = tmp_path / "output"
    assert export_bundle(_source(tmp_path), output) == 1

    exported = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    assert exported == [
        "anonymous-metrics.json",
        "case-00/A.mp4",
        "case-00/AB-side-by-side-muted.mp4",
        "case-00/B.mp4",
        "review-index.json",
    ]


def test_rejects_path_escape(tmp_path) -> None:
    source = _source(tmp_path)
    review = json.loads((source / "review-index.json").read_text())
    review["cases"][0]["A"] = "../.blind-mapping.json"
    (source / "review-index.json").write_text(json.dumps(review))

    with pytest.raises(ValueError, match="inside the bundle"):
        export_bundle(source, tmp_path / "output")


def test_rejects_in_bundle_path_relabeling(tmp_path) -> None:
    source = _source(tmp_path)
    review = json.loads((source / "review-index.json").read_text())
    review["cases"][0]["A"] = ".blind-mapping.json"
    (source / "review-index.json").write_text(json.dumps(review))

    with pytest.raises(ValueError, match=r"must be exactly case-00/A\.mp4"):
        export_bundle(source, tmp_path / "output")


def test_rejects_media_symlink(tmp_path) -> None:
    source = _source(tmp_path)
    media = source / "case-00/A.mp4"
    media.unlink()
    media.symlink_to(source / ".blind-mapping.json")

    with pytest.raises(FileNotFoundError, match="media is missing"):
        export_bundle(source, tmp_path / "output")


def test_rejects_reordered_or_duplicate_case_indices(tmp_path) -> None:
    source = _source(tmp_path)
    review = json.loads((source / "review-index.json").read_text())
    review["cases"][0]["index"] = 1
    (source / "review-index.json").write_text(json.dumps(review))

    with pytest.raises(ValueError, match="contiguous and ordered"):
        export_bundle(source, tmp_path / "output")


def test_rejects_nonempty_output_that_might_contain_mapping(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / ".blind-mapping.json").write_text("secret")

    with pytest.raises(ValueError, match="absent or empty"):
        export_bundle(_source(tmp_path), output)
