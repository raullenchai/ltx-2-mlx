"""Content identity helpers for immutable model-package artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    """Hash a regular artifact without loading it into memory."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transformer_sha256(path: str | Path) -> str:
    """Return transformer content identity, using an immutable HF blob OID.

    Hugging Face snapshots symlink LFS objects to ``blobs/<sha256>``. Reading
    that already-verified object name avoids hashing tens of gigabytes during
    every pipeline construction. Plain local files fall back to streaming
    SHA-256.
    """
    source = Path(path)
    resolved = source.resolve(strict=True)
    if source.is_symlink() and resolved.parent.name == "blobs" and _SHA256_RE.fullmatch(resolved.name):
        return resolved.name
    return file_sha256(source)
