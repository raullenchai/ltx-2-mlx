import json
from pathlib import Path

import pytest

from scripts.split_stage1_manifest_dataset import STAGE1_SOURCES, split_stage1_dataset

PRODUCT_MANIFEST = Path("packages/ltx-trainer/configs/stage1_product_grid_manifest.jsonl")


def _capture(root: Path, index: int) -> None:
    for source in STAGE1_SOURCES:
        path = root / ".precomputed" / source / f"latent_{index:04d}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{source}-{index}".encode())


def test_split_stage1_dataset_hardlinks_complete_prompt_disjoint_views(tmp_path: Path) -> None:
    captured = tmp_path / "captured"
    for index in (4, 9):
        _capture(captured, index)
    records = [
        {"index": 4, "split": "train", "bucket": "product", "prompt": "one"},
        {"index": 9, "split": "validation", "bucket": "product", "prompt": "two"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))

    counts = split_stage1_dataset(manifest, captured, tmp_path / "split", bucket="product")

    assert counts == {"train": 1, "validation": 1}
    for record in records:
        split = record["split"]
        index = record["index"]
        for source in STAGE1_SOURCES:
            original = captured / ".precomputed" / source / f"latent_{index:04d}.safetensors"
            linked = tmp_path / "split" / split / ".precomputed" / source / original.name
            assert original.samefile(linked)


def test_split_stage1_dataset_rejects_incomplete_capture(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"index": 1, "split": "train", "prompt": "one"}) + "\n")

    with pytest.raises(FileNotFoundError, match="missing captured component"):
        split_stage1_dataset(manifest, tmp_path / "captured", tmp_path / "split")


def test_product_grid_manifest_is_prompt_disjoint_and_matches_runtime_shape() -> None:
    records = [json.loads(line) for line in PRODUCT_MANIFEST.read_text().splitlines()]

    assert len(records) == 12
    assert {record["split"] for record in records} == {"train", "validation"}
    assert sum(record["split"] == "train" for record in records) == 8
    assert sum(record["split"] == "validation" for record in records) == 4
    assert len({record["prompt"] for record in records}) == len(records)
    assert len({record["index"] for record in records}) == len(records)
    assert all(
        (record["width"], record["height"], record["frames"], record["frame_rate"])
        == (768, 512, 241, 24.0)
        for record in records
    )
