"""Real pycolmap reconstructions throughout, same style as
test_export_postshot.py -- this module's job is matching real on-disk
diagnostics against real recorded stats, so nothing here is mocked."""

import json
import shutil

import pytest

pytest.importorskip("pycolmap")

import pycolmap

from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db
from vine360.sfm.colmap_adapter import map_and_diagnose
from vine360.sfm.repair_selected_model import repair_selected_model


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


_counter = 0


def _build_real_model(root, *, num_frames):
    global _counter
    _counter += 1
    n = _counter
    database_path = root / "sfm" / f"database-{n}.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = root / f"images-{n}"
    image_dir.mkdir()

    db = pycolmap.Database.open(database_path)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = num_frames
    options.num_points3D = 30 + n  # keep stats distinguishable across builds
    pycolmap.synthesize_dataset(options, db)
    db.close()

    sparse_dir = root / "sfm" / "sparse" / f"scratch-{n}"
    _reconstruction, diagnostics, model_dir = map_and_diagnose(
        database_path, image_dir, sparse_dir, total_images=num_frames
    )
    return model_dir, diagnostics


def _insert_broken_run(conn, run_id, diagnostics):
    """Mimics the real pre-fix bug: selected_model == run_id, an invalid path."""
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES (?, 'h', 'v', '{}', ?, ?, datetime('now'))",
        (run_id, json.dumps(diagnostics.to_dict()), run_id),
    )
    conn.commit()


def test_repair_fixes_a_run_whose_model_is_still_on_disk(project):
    root, conn = project
    model_dir, diagnostics = _build_real_model(root, num_frames=4)
    _insert_broken_run(conn, "sfm-old1", diagnostics)

    result = repair_selected_model(conn, root)

    assert result.fixed == {"sfm-old1": model_dir.relative_to(root).as_posix()}
    assert result.unrecoverable == {}
    row = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = 'sfm-old1'").fetchone()
    assert row[0] == model_dir.relative_to(root).as_posix()
    pycolmap.Reconstruction(root / row[0])  # loads without error


def test_repair_reports_unrecoverable_when_no_disk_model_matches(project):
    root, conn = project
    _build_real_model(root, num_frames=4)  # some unrelated model exists on disk
    from vine360.sfm.colmap_adapter import SfmDiagnostics

    fabricated = SfmDiagnostics(999, 999, 1.0, 1, 99999, 0.0123, 1.0, 1.0)
    _insert_broken_run(conn, "sfm-gone", fabricated)

    result = repair_selected_model(conn, root)

    assert "sfm-gone" in result.unrecoverable
    assert "overwritten" in result.unrecoverable["sfm-gone"]
    row = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = 'sfm-gone'").fetchone()
    assert row[0] == "sfm-gone"  # left untouched, not guessed


def test_repair_leaves_already_correct_rows_alone(project):
    root, conn = project
    model_dir, diagnostics = _build_real_model(root, num_frames=4)
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES ('sfm-good', 'h', 'v', '{}', ?, ?, datetime('now'))",
        (json.dumps(diagnostics.to_dict()), model_dir.relative_to(root).as_posix()),
    )
    conn.commit()

    result = repair_selected_model(conn, root)

    assert result.already_fine == ["sfm-good"]
    assert result.fixed == {}


def test_repair_reports_ambiguous_match_as_unrecoverable(project):
    root, conn = project
    model_dir, diagnostics = _build_real_model(root, num_frames=4)
    duplicate_dir = root / "sfm" / "sparse" / "duplicate"
    shutil.copytree(model_dir, duplicate_dir)
    _insert_broken_run(conn, "sfm-ambiguous", diagnostics)

    result = repair_selected_model(conn, root)

    assert "sfm-ambiguous" in result.unrecoverable
    assert "can't tell them apart" in result.unrecoverable["sfm-ambiguous"]
