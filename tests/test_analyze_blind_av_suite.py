import pytest

from scripts.analyze_blind_av_suite import parse_metrics


def test_parses_ffmpeg_similarity_summaries() -> None:
    result = parse_metrics(
        "SSIM Y:0.84 U:0.97 V:0.98 All:0.889730 (9.57)",
        "PSNR y:27.3 u:40.7 v:41.3 average:28.984457 min:26 max:31",
        "PSNR ch0: 169.207 dB\nPSNR ch1: 169.205 dB",
    )
    assert result == {
        "video_ssim": 0.88973,
        "video_psnr_db": 28.984457,
        "audio_apsnr_db": [169.207, 169.205],
    }


def test_rejects_incomplete_ffmpeg_output() -> None:
    with pytest.raises(ValueError, match="did not contain"):
        parse_metrics("All:0.9", "average:30", "")
