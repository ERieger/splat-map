"""vine360.masking.layers against a real project DB and real files: layers
build independently, the keep-mask is the OR of the enabled ones, and
toggling/excluding/regenerating one layer never touches the others."""

import json
import os
import time

import numpy as np
import pytest
from PIL import Image

from vine360.config import CaptureMode
from vine360.masking.layers import (
    LAYER_EXCLUDED_VIEW,
    LAYER_OVEREXPOSED,
    LAYER_SKY,
    MaskLayerError,
    build_layers_for_frame_set,
    compose_view,
    delete_layer,
    ensure_composites_current,
    find_suspicious_views,
    layer_path,
    layer_summary,
    refresh_edited_from_disk,
    register_legacy_layers,
    render_overlay,
    set_layer_enabled,
    set_view_excluded,
    view_layers,
)
from vine360.project import create_project, open_index_db

KEEP = ("masks", "keep")


def _image(sky_rows=30, clipped=True):
    image = np.full((100, 100, 3), [34, 90, 34], dtype=np.uint8)
    image[:sky_rows] = [135, 206, 235]  # sky-blue band at the top
    if clipped:
        # Clipped patch below the classical sky heuristic's top-60% band
        # (which would otherwise claim it as "sky" -- the very problem the
        # overexposure layer exists for).
        image[65:95, 60:95] = 255
    return image


def _add_view(conn, root, frame_index, face="front", image=None, frame_set="s1"):
    frame_id = f"{frame_set}:{frame_index:06d}"
    rel = f"projections/{frame_id}/{face}.png"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image if image is not None else _image(), "RGB").save(root / rel)
    conn.execute(
        "INSERT OR IGNORE INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, "
        "frame_set_id) VALUES (?, 's1', ?, '{}', 'x', 'y', ?)",
        (frame_id, float(frame_index), frame_set),
    )
    view_id = f"{frame_id}:{face}"
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path) "
        "VALUES (?, ?, 'six-face', 100, 100, '{}', '{}', ?)",
        (view_id, frame_id, rel),
    )
    conn.commit()
    return view_id


def _keep_path(root, view_id):
    frame_id, face = view_id.rsplit(":", 1)
    return root.joinpath(*KEEP, frame_id, f"{face}.png.png")


def _keep(root, view_id):
    return np.asarray(Image.open(_keep_path(root, view_id)))


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def test_each_layer_is_its_own_file_and_row_and_keep_is_their_union(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)

    summary = build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None, LAYER_OVEREXPOSED: {"dilation_px": 0}})

    assert summary.built == {LAYER_SKY: 1, LAYER_OVEREXPOSED: 1}
    assert layer_path(root, view_id, LAYER_SKY).exists()
    assert layer_path(root, view_id, LAYER_OVEREXPOSED).exists()
    layers = {row["layer"]: row for row in view_layers(conn, view_id)}
    assert set(layers) == {LAYER_SKY, LAYER_OVEREXPOSED}
    assert layers[LAYER_OVEREXPOSED]["params"]["clip_threshold"] == 250

    keep = _keep(root, view_id)
    assert keep[10, 50] == 0  # sky
    assert keep[80, 80] == 0  # clipped patch
    assert keep[90, 10] == 255  # ground
    row = conn.execute("SELECT prompts, keep_fraction FROM masks WHERE view_id = ?", (view_id,)).fetchone()
    assert json.loads(row[0]) == [LAYER_OVEREXPOSED, LAYER_SKY]
    assert row[1] == pytest.approx((keep > 127).mean())


def test_regenerating_one_layer_leaves_the_others_untouched(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None, LAYER_OVEREXPOSED: None})
    sky_file = layer_path(root, view_id, LAYER_SKY)
    sky_bytes, sky_mtime = sky_file.read_bytes(), sky_file.stat().st_mtime_ns

    build_layers_for_frame_set(conn, root, "s1", {LAYER_OVEREXPOSED: {"clip_threshold": 256}})  # nothing clips now

    assert sky_file.read_bytes() == sky_bytes and sky_file.stat().st_mtime_ns == sky_mtime
    keep = _keep(root, view_id)
    assert keep[10, 50] == 0, "sky still merged"
    assert keep[80, 80] == 255, "overexposure regenerated empty"


def test_disabling_a_layer_recomposes_and_makes_the_composite_newer(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    before = conn.execute("SELECT updated_at FROM masks WHERE view_id = ?", (view_id,)).fetchone()[0]
    time.sleep(0.01)

    assert set_layer_enabled(conn, root, [view_id], LAYER_SKY, False) == 1

    assert _keep(root, view_id)[10, 50] == 255
    row = conn.execute("SELECT keep_fraction, updated_at FROM masks WHERE view_id = ?", (view_id,)).fetchone()
    assert row[0] == 1.0
    assert row[1] > before, "Pose staleness detection keys off masks.updated_at"
    assert layer_path(root, view_id, LAYER_SKY).exists(), "disabled, not deleted"

    set_layer_enabled(conn, root, [view_id], LAYER_SKY, True)
    assert _keep(root, view_id)[10, 50] == 0


def test_set_layer_enabled_reports_counted_progress(project):
    # The GUI's set-wide "merged" toggle runs this on a background thread;
    # without per-view progress the app looked frozen for the whole set.
    root, conn = project
    view_ids = [_add_view(conn, root, i) for i in range(3)]
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    calls = []

    changed = set_layer_enabled(
        conn, root, view_ids, LAYER_SKY, False, progress_callback=lambda *a: calls.append(a)
    )

    assert changed == 3
    assert [(c[1], c[2]) for c in calls] == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert "view 1/3" in calls[0][0]
    assert "3/3 views" in calls[-1][0]


def test_rebuilding_keeps_a_layer_disabled(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    set_layer_enabled(conn, root, [view_id], LAYER_SKY, False)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    assert view_layers(conn, view_id)[0]["enabled"] is False


def test_excluding_a_view_and_restoring_it(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})

    assert set_view_excluded(conn, root, view_id, True) == 0.0
    assert not _keep(root, view_id).any()

    kf = set_view_excluded(conn, root, view_id, False)
    assert 0.0 < kf < 1.0
    assert not layer_path(root, view_id, LAYER_EXCLUDED_VIEW).exists()


def test_excluding_an_unmasked_view_still_produces_a_keep_mask(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    set_view_excluded(conn, root, view_id, True)
    assert not _keep(root, view_id).any()


def test_deleting_the_last_layer_removes_the_composite(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    delete_layer(conn, root, [view_id], LAYER_SKY)
    assert conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0] == 0
    assert not _keep_path(root, view_id).exists()


def _touch_later(path):
    later = time.time() + 30
    os.utime(path, (later, later))


def test_a_layer_edited_on_disk_is_merged_and_protected_from_regeneration(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    path = layer_path(root, view_id, LAYER_SKY)
    edited = np.zeros((100, 100), dtype=np.uint8)
    edited[90:, :] = 255  # someone painted the bottom strip instead
    Image.fromarray(edited, "L").save(path)
    _touch_later(path)

    assert ensure_composites_current(conn, root, "s1") == 1
    keep = _keep(root, view_id)
    assert keep[10, 50] == 255 and keep[95, 50] == 0
    assert view_layers(conn, view_id)[0]["edited"] is True

    summary = build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    assert summary.skipped_edited == {LAYER_SKY: 1}
    assert _keep(root, view_id)[95, 50] == 0, "hand edit survived"

    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None}, overwrite_edited=True)
    assert _keep(root, view_id)[10, 50] == 0
    assert view_layers(conn, view_id)[0]["edited"] is False


def test_ensure_composites_current_is_a_no_op_when_nothing_changed(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    assert ensure_composites_current(conn, root, "s1") == 0
    assert refresh_edited_from_disk(conn, root, [view_id]) == []


def test_ensure_composites_current_restores_a_missing_keep_file(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    _keep_path(root, view_id).unlink()
    assert ensure_composites_current(conn, root) == 1
    assert _keep(root, view_id)[10, 50] == 0


def test_legacy_class_pngs_are_registered_as_layers_without_touching_the_composite(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    # What the pre-layer build left behind: a raw class PNG + keep + one masks row.
    legacy = root / "masks" / "classes" / view_id / "sky(classical).png"
    legacy.parent.mkdir(parents=True)
    sky = np.zeros((100, 100), dtype=np.uint8)
    sky[:30] = 255
    Image.fromarray(sky, "L").save(legacy)
    keep_path = _keep_path(root, view_id)
    keep_path.parent.mkdir(parents=True)
    Image.fromarray(255 - sky, "L").save(keep_path)
    conn.execute(
        "INSERT INTO masks (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited, "
        "flagged_for_review, updated_at) VALUES (?, 'classical-sky', '1', '[\"sky(classical)\"]', '{}', '{}', 0.7, "
        "0, 0, ?)",
        (view_id, "2099-01-01T00:00:00+00:00"),
    )
    conn.commit()

    assert register_legacy_layers(conn, root, "s1") == 1

    assert not legacy.exists()
    assert layer_path(root, view_id, LAYER_SKY).exists()
    assert [r["layer"] for r in view_layers(conn, view_id)] == [LAYER_SKY]
    assert conn.execute("SELECT updated_at FROM masks").fetchone()[0] == "2099-01-01T00:00:00+00:00"
    assert ensure_composites_current(conn, root, "s1") == 0
    assert register_legacy_layers(conn, root, "s1") == 0, "idempotent"


def test_suspicious_views_compare_the_same_face_across_nearby_frames(project):
    root, conn = project
    for i in range(7):
        _add_view(conn, root, i, "front", _image(sky_rows=60 if i == 3 else 20, clipped=False))
        _add_view(conn, root, i, "down", _image(sky_rows=0, clipped=False))
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: {"dilation_px": 0}})

    suspicious = find_suspicious_views(conn, "s1")

    assert [(s.view_id, s.layer) for s in suspicious] == [("s1:000003:front", LAYER_SKY)]
    assert "nearby frames" in suspicious[0].reason


def test_layer_summary_counts_enabled_and_nonempty(project):
    root, conn = project
    a = _add_view(conn, root, 0)
    _add_view(conn, root, 1, image=_image(clipped=False))
    build_layers_for_frame_set(conn, root, "s1", {LAYER_OVEREXPOSED: None})
    set_layer_enabled(conn, root, [a], LAYER_OVEREXPOSED, False)
    assert layer_summary(conn, "s1") == {LAYER_OVEREXPOSED: {"views": 2, "enabled": 1, "nonempty": 1, "edited": 0}}


def test_render_overlay_tints_enabled_layers(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    build_layers_for_frame_set(conn, root, "s1", {LAYER_OVEREXPOSED: {"dilation_px": 0}})
    overlay = render_overlay(conn, root, view_id)
    assert overlay.shape == (100, 100, 3)
    r, g, b = (int(v) for v in overlay[80, 80])
    assert r > g and b > g, "magenta tint over the clipped patch"
    assert tuple(int(v) for v in overlay[90, 10]) == (34, 90, 34)


def test_unknown_layer_raises(project):
    root, conn = project
    _add_view(conn, root, 0)
    with pytest.raises(MaskLayerError):
        build_layers_for_frame_set(conn, root, "s1", {"clouds": None})


def test_compose_without_layers_removes_nothing_and_returns_none(project):
    root, conn = project
    view_id = _add_view(conn, root, 0)
    assert compose_view(conn, root, view_id) is None
    assert conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0] == 0
