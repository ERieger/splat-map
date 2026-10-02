"""vine360.data_manager: every delete removes exactly its target plus what's
derived from it -- rows and files -- and leaves sibling frame sets,
upstream data and anything outside exports/ alone (docs/adr/0035)."""

import json

import numpy as np
import pytest
from PIL import Image

from vine360.config import CaptureMode
from vine360.data_manager import (
    DataManagerError,
    delete_export_dir,
    delete_frame_set,
    delete_masks_for_frame_set,
    delete_sfm_run,
    delete_views_for_frame_set,
    export_dir_size,
    frame_set_disk_usage,
    list_export_dirs,
    list_frame_sets,
    list_sfm_runs,
    list_sources,
)
from vine360.masking.build import build_mask_for_view
from vine360.project import create_project, open_index_db
from vine360.projection.generate import generate_views_for_frame


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _add_frame_set(conn, root, source_id: str, frame_set_id: str, label: str, n_frames: int = 2) -> list[str]:
    """A real equirectangular frame set on disk, shaped exactly like
    extract_frames writes one."""
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, path, checksum, media_type, projection, width, height, "
        "timestamps, capture_group) VALUES (?, ?, 'x', 'video', 'equirectangular', 128, 64, "
        "'{\"duration_seconds\": 4.0}', NULL)",
        (source_id, f"/media/{source_id}.mp4"),
    )
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
        "VALUES (?, ?, ?, '{}', '2026-01-01')",
        (frame_set_id, source_id, label),
    )
    rng = np.random.default_rng(0)
    frame_ids = []
    for i in range(n_frames):
        frame_id = f"{frame_set_id}:{i:06d}"
        path = root / "frames" / frame_set_id / f"frame_{i + 1:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, size=(64, 128, 3), dtype=np.uint8), "RGB").save(path)
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, ?, ?, '{}', ?, 'c', ?)",
            (frame_id, source_id, float(i), str(path.relative_to(root)), frame_set_id),
        )
        frame_ids.append(frame_id)
    conn.commit()
    return frame_ids


def _project_and_mask(conn, root, frame_ids: list[str]) -> list[str]:
    view_ids = []
    for frame_id in frame_ids:
        for view in generate_views_for_frame(conn, root, frame_id, face_size=16, face_names=["front"]):
            build_mask_for_view(conn, root, view.view_id)
            view_ids.append(view.view_id)
    return view_ids


def _count(conn, sql, *args) -> int:
    return conn.execute(sql, args).fetchone()[0]


def test_list_sources_and_frame_sets_report_counts(project):
    root, conn = project
    a = _add_frame_set(conn, root, "s1", "s1~i0.5", "every 0.5s")
    _add_frame_set(conn, root, "s1", "s1~i1", "every 1s", n_frames=1)
    _project_and_mask(conn, root, a[:1])

    (source,) = list_sources(conn)
    assert source.name == "s1.mp4" and source.frame_set_count == 2 and source.duration_seconds == 4.0

    sets = {fs.frame_set_id: fs for fs in list_frame_sets(conn)}
    assert (sets["s1~i0.5"].frame_count, sets["s1~i0.5"].view_count, sets["s1~i0.5"].mask_count) == (2, 1, 1)
    assert (sets["s1~i1"].frame_count, sets["s1~i1"].view_count) == (1, 0)
    assert sets["s1~i1"].display_name == "s1.mp4 — every 1s"

    usage = frame_set_disk_usage(conn, root, "s1~i0.5")
    assert usage["frames"] > 0 and usage["projections"] > 0 and usage["masks"] > 0


def test_delete_frame_set_cascades_and_leaves_the_sibling_alone(project):
    root, conn = project
    doomed = _add_frame_set(conn, root, "s1", "s1~i0.5", "every 0.5s")
    kept = _add_frame_set(conn, root, "s1", "s1~i1", "every 1s")
    doomed_views = _project_and_mask(conn, root, doomed)
    kept_views = _project_and_mask(conn, root, kept)

    delete_frame_set(conn, root, "s1~i0.5")

    assert [fs.frame_set_id for fs in list_frame_sets(conn)] == ["s1~i1"]
    assert _count(conn, "SELECT COUNT(*) FROM frames WHERE frame_set_id = 's1~i0.5'") == 0
    assert {r[0] for r in conn.execute("SELECT view_id FROM views")} == set(kept_views)
    assert {r[0] for r in conn.execute("SELECT view_id FROM masks")} == set(kept_views)
    assert {r[0] for r in conn.execute("SELECT DISTINCT view_id FROM mask_layers")} == set(kept_views)
    assert not (root / "frames" / "s1~i0.5").exists()
    for frame_id in doomed:
        assert not (root / "projections" / frame_id).exists()
        assert not (root / "masks" / "keep" / frame_id).exists()
    for view_id in doomed_views:
        assert not (root / "masks" / "classes" / view_id).exists()
    assert (root / "frames" / "s1~i1").exists()
    assert all((root / "projections" / f).exists() for f in kept)
    assert _count(conn, "SELECT COUNT(*) FROM sources") == 1  # never touches the source


def test_delete_views_keeps_frames_and_removes_their_masks(project):
    root, conn = project
    frames = _add_frame_set(conn, root, "s1", "s1~i1", "every 1s")
    view_ids = _project_and_mask(conn, root, frames)

    assert delete_views_for_frame_set(conn, root, "s1~i1") == len(frames)

    assert _count(conn, "SELECT COUNT(*) FROM frames") == len(frames)
    assert _count(conn, "SELECT COUNT(*) FROM views") == 0
    assert _count(conn, "SELECT COUNT(*) FROM masks") == 0
    assert _count(conn, "SELECT COUNT(*) FROM mask_layers") == 0
    assert all(not (root / "projections" / f).exists() for f in frames)
    assert all(not (root / "masks" / "keep" / f).exists() for f in frames)
    assert all(not (root / "masks" / "classes" / v).exists() for v in view_ids)
    assert (root / "frames" / "s1~i1").exists()


def test_delete_masks_keeps_frames_and_views(project):
    root, conn = project
    frames = _add_frame_set(conn, root, "s1", "s1~i1", "every 1s")
    other = _add_frame_set(conn, root, "s2", "s2~i1", "every 1s", n_frames=1)
    view_ids = _project_and_mask(conn, root, frames)
    other_views = _project_and_mask(conn, root, other)

    assert delete_masks_for_frame_set(conn, root, "s1~i1") == len(view_ids)

    assert _count(conn, "SELECT COUNT(*) FROM views") == len(view_ids) + len(other_views)
    assert {r[0] for r in conn.execute("SELECT view_id FROM masks")} == set(other_views)
    assert {r[0] for r in conn.execute("SELECT DISTINCT view_id FROM mask_layers")} == set(other_views)
    assert all((root / "projections" / f).exists() for f in frames)
    assert all(not (root / "masks" / "keep" / f).exists() for f in frames)
    assert all(not (root / "masks" / "classes" / v).exists() for v in view_ids)
    assert (root / "masks" / "keep" / other[0]).exists()


def test_delete_frame_set_unknown_raises(project):
    root, conn = project
    with pytest.raises(DataManagerError):
        delete_frame_set(conn, root, "nope")


def _insert_run(conn, root, run_id: str, config: dict) -> None:
    model_dir = root / "sfm" / "sparse" / run_id / "0"
    model_dir.mkdir(parents=True)
    (model_dir.parent / "database.db").write_bytes(b"db")
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES (?, 'h', 'v', ?, ?, ?, datetime('now'))",
        (
            run_id,
            json.dumps(config),
            json.dumps({"registered_images": 3, "total_images": 4}),
            model_dir.relative_to(root).as_posix(),
        ),
    )
    conn.commit()


def test_list_and_delete_sfm_run(project):
    root, conn = project
    _insert_run(conn, root, "sfm-a", {"image_source": "projections", "frame_set_id": "s1~i1"})
    _insert_run(conn, root, "sfm-b", {"image_source": "frames", "source_id": "legacy-src"})

    runs = {run.run_id: run for run in list_sfm_runs(conn)}
    assert runs["sfm-a"].frame_set_id == "s1~i1"
    assert runs["sfm-b"].frame_set_id == "legacy-src"  # pre-frame-set run: its source_id is its frame set
    assert (runs["sfm-a"].registered_images, runs["sfm-a"].total_images) == (3, 4)

    delete_sfm_run(conn, root, "sfm-a")
    assert [run.run_id for run in list_sfm_runs(conn)] == ["sfm-b"]
    assert not (root / "sfm" / "sparse" / "sfm-a").exists()
    assert (root / "sfm" / "sparse" / "sfm-b").exists()

    with pytest.raises(DataManagerError):
        delete_sfm_run(conn, root, "sfm-a")


def test_list_and_delete_export_dirs(project):
    root, _conn = project
    for rel in ("exports/all/postshot", "exports/all/realityscan", "exports/clip_i1/colmap", "exports/old/images"):
        (root / rel).mkdir(parents=True)
        (root / rel / "f.txt").write_text("x")

    listed = {(info.relative_path, info.legacy_layout) for info in list_export_dirs(root)}
    assert listed == {
        ("exports/all/postshot", False),
        ("exports/all/realityscan", False),
        ("exports/clip_i1/colmap", False),
        ("exports/old", True),
    }

    delete_export_dir(root, "exports/clip_i1/colmap")
    assert not (root / "exports" / "clip_i1").exists()  # now-empty capture folder removed too
    delete_export_dir(root, "exports/all/postshot")
    assert (root / "exports" / "all" / "realityscan").exists()


@pytest.mark.parametrize("bad", ["exports", "frames", "exports/../frames", "/etc"])
def test_delete_export_dir_refuses_anything_outside_exports(project, bad):
    root, _conn = project
    with pytest.raises(DataManagerError):
        delete_export_dir(root, bad)
    assert (root / "frames").exists()


def test_export_written_directly_into_exports_is_listed_sized_and_deletable(project):
    """With exports/ itself chosen as the output folder, an export's
    images/masks sit straight in exports/ -- listed as one "exports"
    entry, sized and deleted without touching the <capture>/ folders
    beside it."""
    root, _conn = project
    exports = root / "exports"
    (exports / "images").mkdir()
    (exports / "images" / "a.png").write_bytes(b"x" * 10)
    (exports / "masks").mkdir()
    (exports / "masks" / "a.png").write_bytes(b"x" * 5)
    (exports / "all" / "postshot").mkdir(parents=True)
    (exports / "all" / "postshot" / "b.png").write_bytes(b"x" * 100)

    listed = list_export_dirs(root)
    assert [(i.relative_path, i.legacy_layout, i.in_exports_root) for i in listed] == [
        ("exports", False, True),
        ("exports/all/postshot", False, False),
    ]
    assert export_dir_size(root, listed[0]) == 15  # not the 100 bytes beside it
    assert export_dir_size(root, listed[1]) == 100

    delete_export_dir(root, "exports")
    assert not (exports / "images").exists()
    assert not (exports / "masks").exists()
    assert (exports / "all" / "postshot" / "b.png").exists()
    assert [i.relative_path for i in list_export_dirs(root)] == ["exports/all/postshot"]
