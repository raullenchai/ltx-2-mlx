from __future__ import annotations

import json
from collections import Counter

from scripts.build_stage2_scale_manifest import build_manifest
from scripts.split_stage2_manifest_dataset import LATENT_SOURCES, split_dataset


def test_scale_manifest_has_frozen_split_and_bucket_counts() -> None:
    records = build_manifest()

    assert len(records) == 100
    assert Counter(record["split"] for record in records) == {"train": 80, "validation": 20}
    assert Counter(record["bucket"] for record in records) == {"468": 60, "1536": 30, "3072": 10}
    assert [record["index"] for record in records] == list(range(100))
    assert len({record["seed"] for record in records}) == 100


def test_scale_manifest_is_prompt_disjoint_and_bucket_dimensions_are_exact() -> None:
    records = build_manifest()
    train_prompts = {record["prompt"] for record in records if record["split"] == "train"}
    validation_prompts = {record["prompt"] for record in records if record["split"] == "validation"}

    assert len(train_prompts) == 40
    assert len(validation_prompts) == 10
    assert train_prompts.isdisjoint(validation_prompts)
    for record in records:
        latent_frames = (record["frames"] - 1) // 8 + 1
        tokens = latent_frames * (record["height"] // 32) * (record["width"] // 32)
        assert tokens == int(record["bucket"])


def test_manifest_splitter_builds_complete_hardlinked_views(tmp_path) -> None:
    data = tmp_path / "captured" / ".precomputed"
    records = [
        {"index": 0, "split": "train"},
        {"index": 1, "split": "validation"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    for index in range(2):
        for source in LATENT_SOURCES:
            path = data / source / f"latent_{index:04d}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"latent")
        path = data / "conditions" / f"condition_{index:04d}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"condition")

    counts = split_dataset(manifest, tmp_path / "captured", tmp_path / "splits")

    assert counts == {"train": 1, "validation": 1}
    train_source = data / LATENT_SOURCES[0] / "latent_0000.safetensors"
    train_view = tmp_path / "splits/train/.precomputed" / LATENT_SOURCES[0] / train_source.name
    assert train_source.stat().st_ino == train_view.stat().st_ino
    assert len((tmp_path / "splits/validation/manifest.jsonl").read_text().splitlines()) == 1


def test_manifest_splitter_can_filter_one_bucket(tmp_path) -> None:
    data = tmp_path / "captured" / ".precomputed"
    records = [
        {"index": 0, "split": "train", "bucket": "468"},
        {"index": 1, "split": "train", "bucket": "1536"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    for index in range(2):
        for source in LATENT_SOURCES:
            path = data / source / f"latent_{index:04d}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"latent")
        path = data / "conditions" / f"condition_{index:04d}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"condition")

    counts = split_dataset(
        manifest,
        tmp_path / "captured",
        tmp_path / "bucket",
        bucket="1536",
    )

    assert counts == {"train": 1}
    assert not (tmp_path / "bucket/train/.precomputed" / LATENT_SOURCES[0] / "latent_0000.safetensors").exists()
    assert (tmp_path / "bucket/train/.precomputed" / LATENT_SOURCES[0] / "latent_0001.safetensors").exists()
