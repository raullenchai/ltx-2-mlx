from __future__ import annotations

import hashlib

from ltx_core_mlx.loader.integrity import file_sha256, transformer_sha256


def test_hashes_plain_transformer_content(tmp_path) -> None:
    transformer = tmp_path / "transformer.safetensors"
    transformer.write_bytes(b"weights")

    expected = hashlib.sha256(b"weights").hexdigest()
    assert file_sha256(transformer) == expected
    assert transformer_sha256(transformer) == expected


def test_uses_huggingface_blob_oid_for_snapshot_symlink(tmp_path) -> None:
    digest = "a" * 64
    blob = tmp_path / "models--example--model" / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"placeholder; cache integrity owns content verification")
    snapshot = tmp_path / "models--example--model" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    transformer = snapshot / "transformer.safetensors"
    transformer.symlink_to(blob)

    assert transformer_sha256(transformer) == digest
