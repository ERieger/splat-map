import json

from vine360.ingest.sources import add_source, get_source
from vine360.project import create_project, open_index_db
from vine360.config import CaptureMode
from vine360.runners.base import Runner, RunResult

_FFPROBE_JSON = json.dumps(
    {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 5760,
                "height": 2880,
                "avg_frame_rate": "30/1",
                "duration": "10.0",
            }
        ],
        "format": {"duration": "10.0", "tags": {}},
    }
)


class FakeRunner(Runner):
    def run(self, command, *, cwd=None, env=None, timeout=None):
        return RunResult(command=list(command), returncode=0, stdout=_FFPROBE_JSON, stderr="", duration_seconds=0.0)


def test_add_source_records_capture_group(tmp_path):
    project_root = tmp_path / "proj"
    create_project(project_root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)

    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"not a real video, ffprobe is faked")

    source = add_source(
        conn,
        media_file,
        FakeRunner(),
        capture_group="antigravity-a1-aerial",
        added_at="2026-09-19T00:00:00+00:00",
    )
    assert source.capture_group == "antigravity-a1-aerial"

    reloaded = get_source(conn, source.source_id)
    assert reloaded.capture_group == "antigravity-a1-aerial"
    conn.close()


def test_add_source_capture_group_defaults_to_none(tmp_path):
    project_root = tmp_path / "proj"
    create_project(project_root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)

    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"not a real video, ffprobe is faked")

    source = add_source(conn, media_file, FakeRunner(), added_at="2026-09-19T00:00:00+00:00")
    assert source.capture_group is None
    conn.close()
