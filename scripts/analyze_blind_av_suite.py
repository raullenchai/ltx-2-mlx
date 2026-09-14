#!/usr/bin/env python3
"""Measure anonymous A/B similarity without reading the hidden blind mapping."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

_SSIM_RE = re.compile(r"\bAll:(?P<value>[0-9.]+)")
_PSNR_RE = re.compile(r"\baverage:(?P<value>[0-9.]+)")
_APSNR_RE = re.compile(r"\bPSNR ch(?P<channel>\d+):\s*(?P<value>[0-9.]+) dB")


def parse_metrics(ssim_log: str, psnr_log: str, apsnr_log: str) -> dict[str, object]:
    ssim = _SSIM_RE.search(ssim_log)
    psnr = _PSNR_RE.search(psnr_log)
    audio = {int(match["channel"]): float(match["value"]) for match in _APSNR_RE.finditer(apsnr_log)}
    if ssim is None or psnr is None or not audio:
        raise ValueError("ffmpeg output did not contain SSIM, PSNR, and per-channel APSNR")
    return {
        "video_ssim": float(ssim["value"]),
        "video_psnr_db": float(psnr["value"]),
        "audio_apsnr_db": [audio[channel] for channel in sorted(audio)],
    }


def _ffmpeg_metric(a: Path, b: Path, filter_name: str, *, audio: bool = False) -> str:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(a),
        "-i",
        str(b),
        "-filter_complex",
        f"[0:{'a' if audio else 'v'}][1:{'a' if audio else 'v'}]{filter_name}",
        "-vn" if audio else "-an",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = []
    for case_dir in sorted(args.blind_dir.glob("case-*")):
        a, b = case_dir / "A.mp4", case_dir / "B.mp4"
        if not a.is_file() or not b.is_file():
            continue
        metrics = parse_metrics(
            _ffmpeg_metric(a, b, "ssim"),
            _ffmpeg_metric(a, b, "psnr"),
            _ffmpeg_metric(a, b, "apsnr", audio=True),
        )
        cases.append({"case": case_dir.name, **metrics})
    if not cases:
        raise ValueError("blind directory contains no complete A/B cases")

    channels = [value for case in cases for value in case["audio_apsnr_db"]]
    result = {
        "blinded": True,
        "mapping_read": False,
        "cases": cases,
        "mean_video_ssim": sum(case["video_ssim"] for case in cases) / len(cases),
        "mean_video_psnr_db": sum(case["video_psnr_db"] for case in cases) / len(cases),
        "mean_audio_apsnr_db": sum(channels) / len(channels),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
