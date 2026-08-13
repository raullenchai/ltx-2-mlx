#!/usr/bin/env python3
"""Convert the official LTX-2.5 single-file checkpoints to the split MLX layout.

Reference spec: ``docs/ltx25-port-spec.md`` (§7 "Empaquetado propuesto"). The
pipeline (``ltx-pipelines-mlx``) expects the same model-dir layout as the
LTX-2.3 packs (``dgrauet/ltx-2.3-mlx-q8``), with these 2.5 deltas:

- ``text_encoder/`` subdir with a converted gemma4 mlx-lm checkpoint
  (``config.json`` ``model_type: "gemma4"`` + embedded ``text_config``,
  ``model.safetensors`` with ``language_model.model.*`` keys, and the
  tokenizer extracted from the official file).
- ``keyframes_abs_pos_embedding`` kept inside ``transformer-*.safetensors``
  (the 2.5 DiT config sets ``use_keyframes_abs_pos_embedding: true``).
- The official single-file splits in two: connector weights
  (``*_embeddings_connector.*`` + ``text_embedding_projection.*``) go to
  ``connector.safetensors``; the rest stays in the transformer files.
- ``duration_head.safetensors`` (optional).

Source files (official ``Lightricks/LTX-2.5`` or the ungate mirror
``dummy9996/LTX-2.5-22b-ungate`` — identical sha256):

    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors   (42 GB)
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors         (42 GB)
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors           (26 GB)
    vae/ltx-2.5-video-vae-conv-bf16.safetensors                           (1.4 GB)
    vae/ltx-2.5-audio-vae-bf16.safetensors                                 (365 MB)
    model_patches/ltx-2.5-duration-head-bf16.safetensors                   (3.8 MB)
    latent_upscale_models/ltx-2.5-spatial-upscaler-x2-bf16-1.0.safetensors
    latent_upscale_models/ltx-2.5-temporal-upscaler-x2-bf16-1.0.safetensors

Each component converts independently (``--step``), so you never need all
source files on disk at once. Run from the repo root with the project venv:

    uv run --project ltx-2-mlx python scripts/convert_ltx25_to_mlx.py \\
        --out models/ltx-2.5-mlx \\
        --step transformer-distilled --distilled <path> --q8
    uv run --project ltx-2-mlx python scripts/convert_ltx25_to_mlx.py \\
        --out models/ltx-2.5-mlx --step connector \\
        --distilled <path> --gemma4 <path>
    uv run --project ltx-2-mlx python scripts/convert_ltx25_to_mlx.py \\
        --out models/ltx-2.5-mlx --step text-encoder --gemma4 <path> [--gemma4-bits 4]
    ... etc for vae / audio-vae / duration-head / upscalers / config

Memory notes:
- ``transformer-distilled``/``transformer-dev`` with ``--q8`` peak ~13 GB
  (weights are quantized block-by-block; only one bf16 block is resident at
  a time). Without ``--q8`` the bf16 output needs ~42 GB of free RAM/disk.
- ``text-encoder`` bf16 pass-through needs ~26 GB; ``--gemma4-bits 4``
  quantizes in-memory (peak ~30 GB) and shrinks the file to ~7 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSFORMER_PREFIX = "model.diffusion_model."
# Keys that belong to connector.safetensors, not the transformer file.
CONNECTOR_KEYS = ("video_embeddings_connector.", "audio_embeddings_connector.")

# 2.3-compatible renames applied to the stripped transformer keys.
_RENAMES = (
    ("adaln_single.emb.timestep_embedder.linear_1.", "adaln_single.emb.timestep_embedder.linear1."),
    ("audio_adaln_single.emb.timestep_embedder.linear_1.", "audio_adaln_single.emb.timestep_embedder.linear1."),
    ("prompt_adaln_single.emb.timestep_embedder.linear_1.", "prompt_adaln_single.emb.timestep_embedder.linear1."),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_1.", "audio_prompt_adaln_single.emb.timestep_embedder.linear1."),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_1.", "av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear1."),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_1.", "av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear1."),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.", "av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear1."),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_1.", "av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear1."),
    # Same for linear_2 (the MLX adaln embedders use linear1/linear2).
    ("timestep_embedder.linear_1.", "timestep_embedder.linear1."),
    ("timestep_embedder.linear_2.", "timestep_embedder.linear2."),
    # Official 2.5 single-files use Comfy-style names; the MLX blocks (2.3
    # layout) use the flat forms. (No leading dot: ``audio_ff.net...`` is
    # preceded by ``_``; ``ff.net...`` by nothing.)
    (".to_out.0.", ".to_out."),
    ("ff.net.0.proj.", "ff.proj_in."),
    ("ff.net.2.", "ff.proj_out."),
)


def rename_2_3(name: str) -> str:
    """Apply the ``linear_1`` → ``linear1`` rename (matches the 2.3 convert)."""
    for old, new in _RENAMES:
        name = name.replace(old, new)
    return name


def _to_mlx_conv_layout(tensor: mx.array, *, transposed: bool = False) -> mx.array:
    """Permute official 2.5 conv weights (PyTorch layout) to the MLX layout.

    The official LTX-2.5 files ship conv kernels in PyTorch layout, but the
    MLX port (2.3 pack + ``nn.Conv{1,2,3}d``) uses ``(out, *kernel, in)``:

    - Conv3d         (O, I, D, H, W) -> (O, D, H, W, I)
    - Conv2d         (O, I, H, W)    -> (O, H, W, I)
    - Conv1d         (O, I, L)       -> (O, L, I)
    - ConvTranspose1d (I, O, L)      -> (O, L, I)  (``transposed=True``, e.g. vocoder ``ups.N.weight``)
    """
    if tensor.ndim == 5:
        return tensor.transpose(0, 2, 3, 4, 1)
    if tensor.ndim == 4:
        return tensor.transpose(0, 2, 3, 1)
    if tensor.ndim == 3:
        return tensor.transpose(1, 2, 0) if transposed else tensor.transpose(0, 2, 1)
    return tensor


def read_metadata(path: Path) -> dict:
    """Read the ``__metadata__`` dict from a safetensors file header."""
    with safe_open(str(path), framework="numpy") as f:
        return dict(f.metadata() or {})


_DTYPE_TO_NP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


# -- raw safetensors reader -------------------------------------------------
# safetensors' numpy backend cannot materialize BF16 tensors ("data type
# 'bfloat16' not understood"), so tensors are read directly: parse the JSON
# header once, then seek+read the raw bytes of each tensor (streaming, one
# tensor at a time — the blockwise transformer conversion stays low-RAM).


@contextmanager
def _st_open(path: Path):
    """Open a safetensors file; yield ``(fh, header, data_start, keys)``."""
    import struct

    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        keys = [k for k in header if k != "__metadata__"]
        yield fh, header, 8 + hlen, keys


def _st_tensor(fh, header, data_start: int, key: str) -> mx.array:
    """Read one tensor's raw bytes from an open handle as ``mx.array``."""
    info = header[key]
    start, end = info["data_offsets"]
    fh.seek(data_start + start)
    buf = fh.read(end - start)
    dtype = info["dtype"]
    shape = info["shape"]
    if dtype == "BF16":
        arr = mx.array(np.frombuffer(buf, dtype=np.uint16).reshape(shape))
        return arr.view(mx.bfloat16)
    return mx.array(np.frombuffer(buf, dtype=_DTYPE_TO_NP[dtype]).reshape(shape))


def _bytes_tensor(path: Path, key: str) -> bytes:
    with safe_open(str(path), framework="numpy") as f:
        return bytes(f.get_tensor(key).tolist())


def _save_mx_arrays(path: Path, arrays: dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), arrays)
    print(f"  wrote {path} ({sum(a.nbytes for a in arrays.values()) / 1e9:.2f} GB, {len(arrays)} tensors)")


# ---------------------------------------------------------------------------
# Step: config.json + embedded_config.json (transformer config)
# ---------------------------------------------------------------------------


def convert_config(out_dir: Path, transformer_file: Path) -> None:
    """Write ``config.json`` / ``embedded_config.json`` from the single-file metadata.

    Layout matches the 2.3 packs: a flat dict (``model_version``, model
    hyperparameters). The 2.5 embedded config is a JSON string under
    ``__metadata__.config`` with a ``{"transformer": {...}}`` envelope.
    """
    meta = read_metadata(transformer_file)
    raw = meta.get("config", "")
    if not raw:
        raise ValueError(f"no __metadata__.config in {transformer_file}")
    embedded = json.loads(raw)  # {"transformer": {...}}
    transformer = embedded.get("transformer", embedded)

    flat = dict(transformer)
    flat["model_version"] = meta.get("model_version", "2.5.0")
    flat["is_v2"] = True
    flat["model_type"] = "AudioVideo"
    # Keep the nested envelope too (both shapes are accepted by the loaders).
    embedded_flat = {"transformer": flat}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("config.json", flat), ("embedded_config.json", embedded_flat)):
        (out_dir / name).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out_dir / name}")


# ---------------------------------------------------------------------------
# Step: transformer-distilled / transformer-dev
# ---------------------------------------------------------------------------


def convert_transformer(
    src: Path,
    out_dir: Path,
    out_name: str,
    q8: bool,
) -> None:
    """Convert one DiT single-file to ``transformer-*.safetensors``.

    Strips ``model.diffusion_model.`` → ``transformer.``, excludes the
    connector weights (they go to ``connector.safetensors``), renames
    ``linear_1`` → ``linear1``, keeps ``keyframes_abs_pos_embedding``.
    With ``--q8``, transformer blocks are quantized (group 64, 8-bit)
    block-by-block so only one bf16 block is resident at a time.
    """
    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig

    meta = read_metadata(src)
    raw_config = meta.get("config", "")
    if not raw_config:
        raise ValueError(f"no __metadata__.config in {src}")
    transformer_config = json.loads(raw_config)["transformer"]
    config = LTXModelConfig.from_checkpoint_config(transformer_config)
    dit = LTXModel(config)
    if not config.use_keyframes_abs_pos_embedding:
        print("  warning: checkpoint config does not set use_keyframes_abs_pos_embedding", file=sys.stderr)

    weights: dict[str, mx.array] = {}
    block_re = re.compile(r"^transformer_blocks\.(\d+)\.")

    with _st_open(src) as (fh, header, data_start, keys):
        # 1. Non-block weights (small; loaded in one pass).
        non_block: dict[str, mx.array] = {}
        for key in keys:
            if not key.startswith(TRANSFORMER_PREFIX):
                continue
            name = rename_2_3(key[len(TRANSFORMER_PREFIX) :])
            if name.startswith(CONNECTOR_KEYS):
                continue
            if block_re.match(name):
                continue
            non_block[name] = _st_tensor(fh, header, data_start, key)
        print(f"  loaded {len(non_block)} non-block tensors")
        dit.load_weights(list(non_block.items()), strict=False)
        weights.update(non_block)
        del non_block

        # 2. Transformer blocks one at a time (quantize before the next load).
        num_layers = config.num_layers
        for block_idx in range(num_layers):
            prefix = f"transformer_blocks.{block_idx}."
            block = dit.transformer_blocks[block_idx]
            bf16: dict[str, mx.array] = {}
            for key in keys:
                if not key.startswith(TRANSFORMER_PREFIX):
                    continue
                name = rename_2_3(key[len(TRANSFORMER_PREFIX) :])
                if name.startswith(prefix):
                    bf16[name[len(prefix) :]] = _st_tensor(fh, header, data_start, key)
            if not bf16:
                raise ValueError(f"no weights found for {prefix} in {src}")
            # Video ff in the official 2.5 file has no bias (ff_bias=false):
            # synthesize zeros (equivalent to bias=None in the linear layer).
            if "ff.proj_in.weight" in bf16 and "ff.proj_in.bias" not in bf16:
                bf16["ff.proj_in.bias"] = mx.zeros(block.ff.proj_in.bias.shape, dtype=mx.bfloat16)
                bf16["ff.proj_out.bias"] = mx.zeros(block.ff.proj_out.bias.shape, dtype=mx.bfloat16)
            block.load_weights(list(bf16.items()), strict=True)
            del bf16
            if q8:
                nn.quantize(
                    block,
                    group_size=64,
                    bits=8,
                    class_predicate=lambda _p, m: isinstance(m, nn.Linear),
                )
            from mlx.utils import tree_flatten

            for name, arr in tree_flatten(block.parameters()):
                weights[f"transformer_blocks.{block_idx}.{name}"] = arr
            if block_idx % 8 == 0:
                print(f"  block {block_idx}/{num_layers} done")

    mx.eval(*weights.values())
    out = out_dir / out_name
    _save_mx_arrays(out, {f"transformer.{k}": v for k, v in weights.items()})


# ---------------------------------------------------------------------------
# Step: connector.safetensors
# ---------------------------------------------------------------------------


def convert_connector(out_dir: Path, transformer_file: Path, te_file: Path) -> None:
    """Build ``connector.safetensors`` (prefix ``connector.``).

    Sources:
    - ``*_embeddings_connector.*`` from the DiT single-file
      (``model.diffusion_model.`` stripped);
    - ``text_embedding_projection.*`` from the gemma4 TE file.
    """
    weights: dict[str, mx.array] = {}
    with _st_open(transformer_file) as (fh, header, data_start, keys):
        for key in keys:
            if not key.startswith(TRANSFORMER_PREFIX):
                continue
            name = key[len(TRANSFORMER_PREFIX) :]
            if name.startswith(CONNECTOR_KEYS):
                weights[f"connector.{name}"] = _st_tensor(fh, header, data_start, key)
    with _st_open(te_file) as (fh, header, data_start, keys):
        for key in keys:
            if key.startswith("text_embedding_projection."):
                weights[f"connector.{key}"] = _st_tensor(fh, header, data_start, key)
    if not weights:
        raise ValueError("no connector tensors found (check --distilled/--dev and --gemma4 paths)")
    mx.eval(*weights.values())
    _save_mx_arrays(out_dir / "connector.safetensors", weights)


# ---------------------------------------------------------------------------
# Step: text-encoder (gemma4 mlx-lm dir)
# ---------------------------------------------------------------------------


def convert_text_encoder(te_file: Path, out_dir: Path, gemma4_bits: int | None) -> None:
    """Convert the gemma4 TE single-file into ``<out>/text_encoder/``.

    Layout (mlx-lm 0.31.3 ``model_type: "gemma4"``):
    - ``config.json``: ``{"model_type": "gemma4", "text_config": {...}}``
      (the embedded gemma4 text config; 48 layers, hidden 3840, k_eq_v,
      rope proportional 0.25@1e6 on the 8 full-attention layers).
    - ``model.safetensors``: ``model.*`` keys remapped to
      ``language_model.model.*`` (the gemma4 wrapper's sanitize expects
      HF-style ``language_model.model.`` paths; verified against 0.31.3).
    - ``tokenizer.json`` / ``tokenizer_config.json`` / ``generation_config.json``
      extracted from the ``tokenizer_json`` / ``hf_asset__*`` U8 tensors.
    - With ``--gemma4-bits 4|8``, quantizes the language model in-memory
      (group 64) and records ``quantization`` in config.json.
    """
    te_out = out_dir / "text_encoder"
    te_out.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(te_file)
    try:
        gemma_config = json.loads(meta["gemma_config"])
    except KeyError as exc:
        raise ValueError(f"no __metadata__.gemma_config in {te_file}") from exc
    text_config = gemma_config["text_config"]
    expected_layers = int(os.environ.get("LTX25_CONVERT_EXPECTED_TE_LAYERS", "48"))
    if text_config.get("num_hidden_layers") != expected_layers:
        raise ValueError(
            f"unexpected gemma4 text_config layers: {text_config.get('num_hidden_layers')} "
            f"(expected {expected_layers}; set LTX25_CONVERT_EXPECTED_TE_LAYERS to override)"
        )

    config = {"model_type": "gemma4", "text_config": text_config}

    # Tokenizer assets.
    with safe_open(str(te_file), framework="numpy") as f:
        for key in f.keys():  # noqa: SIM118 -- safe_open handle, not a dict
            if key == "tokenizer_json":
                (te_out / "tokenizer.json").write_bytes(_bytes_tensor(te_file, key))
            elif key.startswith("hf_asset__"):
                asset_name = key[len("hf_asset__") :]
                if asset_name in ("tokenizer_config.json", "generation_config.json"):
                    (te_out / asset_name).write_bytes(_bytes_tensor(te_file, key))

    # Weights: model.* → language_model.model.* (skip non-text groups).
    weights: dict[str, mx.array] = {}
    with _st_open(te_file) as (fh, header, data_start, keys):
        for key in keys:
            if key.startswith("model."):
                weights["language_model.model." + key[len("model.") :]] = _st_tensor(fh, header, data_start, key)
    print(f"  loaded {len(weights)} gemma4 tensors")

    model_path = te_out / "model.safetensors"
    if gemma4_bits:
        # 1. Write the bf16 file + config first so mlx-lm can load this dir,
        #    then quantize in-memory and overwrite both.
        mx.eval(*weights.values())
        _save_mx_arrays(model_path, weights)
        del weights
        (te_out / "config.json").write_text(json.dumps(config, indent=2))

        from mlx_lm.utils import load_model

        model, _ = load_model(te_out, lazy=True)
        nn.quantize(
            model,
            group_size=64,
            bits=gemma4_bits,
            class_predicate=lambda _p, m: isinstance(m, nn.Linear),
        )
        from mlx.utils import tree_flatten

        flat = dict(tree_flatten(model.parameters()))
        mx.eval(*flat.values())
        _save_mx_arrays(model_path, flat)
        config["quantization"] = {"group_size": 64, "bits": gemma4_bits, "mode": "affine"}
    else:
        mx.eval(*weights.values())
        _save_mx_arrays(model_path, weights)
    (te_out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  wrote {te_out / 'config.json'}")


# ---------------------------------------------------------------------------
# Step: vae (conv) / audio-vae / duration-head / upscalers
# ---------------------------------------------------------------------------


def convert_vae_conv(vae_file: Path, out_dir: Path) -> None:
    """Split the 2.5 conv VAE into ``vae_encoder.safetensors`` + ``vae_decoder.safetensors``.

    The 2.5 conv file is architecturally identical to the ported 2.3 VAE.
    ``per_channel_statistics.mean-of-means`` → ``_mean_of_means`` (2.3 pack
    naming; the loader remaps). The official file ships ONE per-channel pair
    shared by encoder and decoder; the decoder receives the same real pair
    under ``per_channel_statistics.mean`` / ``.std`` (F32) so
    ``denormalize_latent`` applies the real statistics. Port-1 synthesized
    identity stats (mean=0/std=1) were removed: they crushed ~31% of decoded
    pixels to black. A warning is only emitted when BOTH the official pair
    and an explicit decoder pair are missing.
    """
    enc: dict[str, mx.array] = {}
    dec: dict[str, mx.array] = {}
    with _st_open(vae_file) as (fh, header, data_start, keys):
        for key in keys:
            tensor = _st_tensor(fh, header, data_start, key)
            if key.endswith(".conv.weight"):
                # Official 2.5 file: PyTorch (O, I, D, H, W); MLX port: (O, D, H, W, I).
                tensor = _to_mlx_conv_layout(tensor)
            if key.startswith("encoder."):
                enc[f"vae_encoder.{key[len('encoder.') :]}"] = tensor
            elif key.startswith("decoder."):
                dec[f"vae_decoder.{key[len('decoder.') :]}"] = tensor
            elif key.startswith("per_channel_statistics."):
                name = key[len("per_channel_statistics.") :]
                if name in ("mean-of-means", "std-of-means"):
                    mlx_name = name.replace("mean-of-means", "_mean_of_means").replace("std-of-means", "_std_of_means")
                    enc[f"vae_encoder.per_channel_statistics.{mlx_name}"] = tensor
                elif name in ("mean", "std"):
                    dec[f"vae_decoder.per_channel_statistics.{name}"] = tensor
    if "vae_decoder.per_channel_statistics.mean" not in dec:
        # The decoder's real stats are the official ``mean-of-means`` pair
        # (already staged on the encoder side). Reuse them so decode
        # denormalization matches training statistics.
        enc_mean_key = "vae_encoder.per_channel_statistics._mean_of_means"
        if enc_mean_key in enc:
            print(
                "  note: decoder per_channel_statistics not in official file; "
                "reusing the real encoder (mean-of-means) pair",
                file=sys.stderr,
            )
            dec["vae_decoder.per_channel_statistics.mean"] = enc[enc_mean_key].astype(mx.float32)
            dec["vae_decoder.per_channel_statistics.std"] = enc[
                "vae_encoder.per_channel_statistics._std_of_means"
            ].astype(mx.float32)
        else:
            print(
                "  warning: decoder per_channel_statistics missing and no official "
                "mean-of-means pair available; falling back to identity "
                "(mean=0/std=1) — expected to crush pixels to black",
                file=sys.stderr,
            )
            dec["vae_decoder.per_channel_statistics.mean"] = mx.zeros((128,), dtype=mx.float32)
            dec["vae_decoder.per_channel_statistics.std"] = mx.ones((128,), dtype=mx.float32)
    _save_mx_arrays(out_dir / "vae_encoder.safetensors", enc)
    _save_mx_arrays(out_dir / "vae_decoder.safetensors", dec)


def convert_audio_vae(audio_vae_file: Path, out_dir: Path) -> None:
    """Split the combined audio file into ``audio_vae.safetensors`` + ``vocoder.safetensors``.

    Same architecture as the 2.3 port; only the weights are new. The
    ``mean-of-means``/``std-of-means`` stats are renamed to the 2.3
    ``_mean_of_means``/``_std_of_means`` layout the loader expects.
    """
    audio: dict[str, mx.array] = {}
    vocoder: dict[str, mx.array] = {}
    with _st_open(audio_vae_file) as (fh, header, data_start, keys):
        for key in keys:
            tensor = _st_tensor(fh, header, data_start, key)
            if key.startswith("vocoder."):
                # Official 2.5 file keys are inconsistently prefixed
                # (``vocoder.X`` or ``vocoder.vocoder.X``); normalize to one.
                name = key
                while name.startswith("vocoder."):
                    name = name[len("vocoder.") :]
                if tensor.ndim == 3 and name.endswith((".weight", ".filter", "_basis")):
                    # Conv1d (O, I, L) -> (O, L, I); ConvTranspose1d ``ups.N``
                    # (I, O, L) -> (O, L, I); lowpass upsample/downsample
                    # filters and STFT basis matrices are 3-D in the same
                    # PyTorch layout.
                    transposed = bool(re.search(r"(^|\.)ups\.\d+\.weight$", name))
                    tensor = _to_mlx_conv_layout(tensor, transposed=transposed)
                vocoder[f"vocoder.{name}"] = tensor
            elif key.startswith("audio_vae."):
                name = key[len("audio_vae.") :]
                if name.startswith("per_channel_statistics."):
                    name = name.replace("mean-of-means", "_mean_of_means").replace("std-of-means", "_std_of_means")
                if name.endswith(".conv.weight"):
                    # Official 2.5 file: PyTorch (O, I, H, W); MLX port: (O, H, W, I).
                    tensor = _to_mlx_conv_layout(tensor)
                audio[f"audio_vae.{name}"] = tensor
    _save_mx_arrays(out_dir / "audio_vae.safetensors", audio)
    _save_mx_arrays(out_dir / "vocoder.safetensors", vocoder)


def convert_duration_head(head_file: Path, out_dir: Path) -> None:
    """Extract ``duration_head.safetensors`` (16 tensors, prefix ``duration_head.``)."""
    weights: dict[str, mx.array] = {}
    with _st_open(head_file) as (fh, header, data_start, keys):
        for key in keys:
            tensor = _st_tensor(fh, header, data_start, key)
            if key.startswith("model.diffusion_model.duration_head."):
                key = key[len("model.diffusion_model.duration_head.") :]
            elif key.startswith("duration_head."):
                key = key[len("duration_head.") :]
            weights[f"duration_head.{key}"] = tensor
    _save_mx_arrays(out_dir / "duration_head.safetensors", weights)


def convert_upscalers(
    out_dir: Path,
    spatial: Path | None,
    temporal: Path | None,
) -> None:
    """Copy the latent upscalers (bare keys) + their ``*_config.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, stem in ((spatial, "spatial_upscaler_x2"), (temporal, "temporal_upscaler_x2")):
        if src is None:
            continue
        meta = read_metadata(src)
        raw_config = meta.get("config", "")
        if raw_config:
            (out_dir / f"{stem}_config.json").write_text(
                json.dumps({"config": json.loads(raw_config)}, indent=2)
            )
            print(f"  wrote {out_dir / f'{stem}_config.json'}")
        weights: dict[str, mx.array] = {}
        with _st_open(src) as (fh, header, data_start, keys):
            for key in keys:
                tensor = _st_tensor(fh, header, data_start, key)
                if key.endswith(".weight") and tensor.ndim >= 4:
                    # Official 2.5 file: PyTorch conv layout (O, I, [D,] H, W);
                    # MLX port: (O, [D,] H, W, I). Covers Conv3d blocks and the
                    # Conv2d ``upsampler.0.weight`` pixel-shuffle conv.
                    tensor = _to_mlx_conv_layout(tensor)
                weights[key] = tensor
        _save_mx_arrays(out_dir / f"{stem}.safetensors", weights)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEPS = (
    "config",
    "transformer-distilled",
    "transformer-dev",
    "connector",
    "text-encoder",
    "vae",
    "audio-vae",
    "duration-head",
    "upscalers",
    "all",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path, help="Output model dir (split MLX layout).")
    parser.add_argument(
        "--step",
        choices=STEPS,
        default="all",
        help="Component to convert (independent per-component steps; default: all).",
    )
    parser.add_argument("--distilled", type=Path, help="distilled transformer single-file.")
    parser.add_argument("--dev", type=Path, help="dev transformer single-file (optional).")
    parser.add_argument("--gemma4", type=Path, help="gemma4-12b-with-proj TE single-file.")
    parser.add_argument("--vae-conv", type=Path, help="ltx-2.5-video-vae-conv single-file.")
    parser.add_argument("--audio-vae", type=Path, help="ltx-2.5-audio-vae single-file.")
    parser.add_argument("--duration-head", type=Path, help="ltx-2.5-duration-head single-file.")
    parser.add_argument("--spatial-upscaler", type=Path, help="2.5 spatial upscaler x2 single-file.")
    parser.add_argument("--temporal-upscaler", type=Path, help="2.5 temporal upscaler x2 single-file.")
    parser.add_argument(
        "--q8",
        action="store_true",
        help="Quantize transformer blocks to 8-bit (group 64), matching the 2.3 q8 packs.",
    )
    parser.add_argument(
        "--gemma4-bits",
        type=int,
        choices=(4, 8),
        default=None,
        help="Quantize the gemma4 text encoder to 4/8-bit (group 64) in-memory.",
    )
    args = parser.parse_args(argv)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def _require(path: Path | None, flag: str, step: str) -> Path:
        if path is None:
            parser.error(f"--step {step} requires {flag}")
        if not path.exists():
            parser.error(f"{flag} file not found: {path}")
        return path

    steps = [args.step] if args.step != "all" else [
        "config",
        "connector",
        "text-encoder",
        "vae",
        "audio-vae",
        "duration-head",
        "upscalers",
    ]

    for step in steps:
        print(f"== step: {step} ==")
        if step == "config":
            convert_config(out_dir, _require(args.distilled or args.dev, "--distilled/--dev", step))
        elif step in ("transformer-distilled", "transformer-dev"):
            src = _require(args.distilled if step == "transformer-distilled" else args.dev, f"--{step.split('-')[1]}", step)
            out_name = "transformer-distilled.safetensors" if step == "transformer-distilled" else "transformer-dev.safetensors"
            convert_transformer(src, out_dir, out_name, q8=args.q8)
        elif step == "connector":
            te = _require(args.gemma4, "--gemma4", step)
            tf = _require(args.distilled or args.dev, "--distilled/--dev", step)
            convert_connector(out_dir, tf, te)
        elif step == "text-encoder":
            convert_text_encoder(_require(args.gemma4, "--gemma4", step), out_dir, args.gemma4_bits)
        elif step == "vae":
            convert_vae_conv(_require(args.vae_conv, "--vae-conv", step), out_dir)
        elif step == "audio-vae":
            convert_audio_vae(_require(args.audio_vae, "--audio-vae", step), out_dir)
        elif step == "duration-head":
            convert_duration_head(_require(args.duration_head, "--duration-head", step), out_dir)
        elif step == "upscalers":
            convert_upscalers(out_dir, args.spatial_upscaler, args.temporal_upscaler)
    print(f"done. Model dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
