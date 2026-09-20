import json

import pytest

from vine360.ingest.sources import SourceError, add_source, get_source, remove_source
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


def test_remove_source_deletes_row_but_not_original_file(tmp_path):
    project_root = tmp_path / "proj"
    create_project(project_root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)

    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"not a real video, ffprobe is faked")
    source = add_source(conn, media_file, FakeRunner(), added_at="2026-09-19T00:00:00+00:00")

    remove_source(conn, project_root, source.source_id)

    with pytest.raises(SourceError):
        get_source(conn, source.source_id)
    assert media_file.exists(), "the original media file must never be deleted"
    conn.close()


def test_remove_source_also_clears_its_frames(tmp_path):
    project_root = tmp_path / "proj"
    create_project(project_root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)

    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"not a real video, ffprobe is faked")
    source = add_source(conn, media_file, FakeRunner(), added_at="2026-09-19T00:00:00+00:00")

    frames_dir = project_root / "frames" / source.source_id
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_000001.png").write_bytes(b"fake")
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES (?, ?, 0.0, '{}', 'frames/x/frame_000001.png', 'deadbeef')",
        (f"{source.source_id}:000000", source.source_id),
    )
    conn.commit()

    remove_source(conn, project_root, source.source_id)

    assert conn.execute("SELECT COUNT(*) FROM frames WHERE source_id = ?", (source.source_id,)).fetchone()[0] == 0
    assert not frames_dir.exists()
    conn.close()


def test_add_source_reports_progress(tmp_path):
    project_root = tmp_path / "proj"
    create_project(project_root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)

    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"not a real video, ffprobe is faked")

    messages = []
    add_source(
        conn,
        media_file,
        FakeRunner(),
        added_at="2026-09-20T00:00:00+00:00",
        progress_callback=messages.append,
    )
    assert any("Probing" in m for m in messages)
    assert any("checksum" in m.lower() for m in messages)
    conn.close()


def test_remove_source_unknown_id_raises(tmp_path):
    project_root = tmp_path / "proj"
    create_project(project_root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)
    with pytest.raises(SourceError):
        remove_source(conn, project_root, "does-not-exist")
    conn.close()
