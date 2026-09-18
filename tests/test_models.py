from vine360.config import CaptureMode, MediaType, Projection
from vine360.models import Frame, Project, Source


def test_project_round_trip_dict():
    project = Project(
        schema_version=1,
        project_id="abc",
        name="Test",
        created_at="2026-01-01T00:00:00+00:00",
        working_crs=None,
        software_versions={"vine360": "0.1.0"},
        capture_mode=CaptureMode.MIXED,
    )
    restored = Project.from_dict(project.to_dict())
    assert restored == project


def test_source_round_trip_dict():
    source = Source(
        source_id="s1",
        path="/media/clip.mp4",
        checksum="deadbeef",
        media_type=MediaType.VIDEO,
        projection=Projection.EQUIRECTANGULAR,
        dimensions=(5760, 2880),
        timestamps={"duration_seconds": 12.5},
        capture_group=None,
    )
    restored = Source.from_dict(source.to_dict())
    assert restored == source


def test_frame_round_trip_dict():
    frame = Frame(
        frame_id="s1:000000",
        source_id="s1",
        source_time=0.0,
        extraction_settings={"mode": "interval", "interval_seconds": 1.0, "requested_count": None},
        path="frames/s1/frame_000001.png",
        checksum="cafef00d",
    )
    restored = Frame.from_dict(frame.to_dict())
    assert restored == frame
