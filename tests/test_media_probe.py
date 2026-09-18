import json
from pathlib import Path

from vine360.ingest.media_probe import build_ffprobe_command, parse_ffprobe_json


def test_build_ffprobe_command():
    cmd = build_ffprobe_command(Path("/media/clip.mp4"))
    assert cmd == [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "/media/clip.mp4",
    ]


FFPROBE_EQUIRECT_VIDEO = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "hevc",
            "width": 5760,
            "height": 2880,
            "avg_frame_rate": "30000/1001",
            "duration": "12.345",
        }
    ],
    "format": {"duration": "12.5", "tags": {"location": "+12.3456-078.9012/"}},
}

FFPROBE_ORDINARY_PHOTO = {
    "streams": [
        {"codec_type": "video", "codec_name": "mjpeg", "width": 4000, "height": 3000, "avg_frame_rate": "0/0"}
    ],
    "format": {"tags": {}},
}


def test_parse_ffprobe_json_detects_2to1_equirectangular_video():
    meta = parse_ffprobe_json(json.dumps(FFPROBE_EQUIRECT_VIDEO))
    assert meta.width == 5760
    assert meta.height == 2880
    assert meta.is_2to1_aspect is True
    assert meta.duration_seconds == 12.5
    assert meta.codec == "hevc"
    assert round(meta.frame_rate, 3) == round(30000 / 1001, 3)
    assert meta.format_tags["location"] == "+12.3456-078.9012/"


def test_parse_ffprobe_json_ordinary_photo_is_not_2to1():
    meta = parse_ffprobe_json(json.dumps(FFPROBE_ORDINARY_PHOTO))
    assert meta.is_2to1_aspect is False
    assert meta.frame_rate is None
    assert meta.duration_seconds is None
