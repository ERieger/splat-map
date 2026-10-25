"""Tag-size tables (docs/adr/0047): the built-in Vine360 kit sheet, the
per-device choice of table, and sizing control points from a table on
import or afterwards."""

from pathlib import Path

import pytest

from vine360.georef.control_points import apply_tag_sizes, import_control_points, load_control_points, update_control_point
from vine360.georef.tag_sizes import (
    BUILTIN_TABLE,
    TagSizeTableError,
    default_table_path,
    read_tag_size_table,
    set_default_table_path,
)
from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db

GCP_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gcp" / "ESTO1_GCPs_23Oct26.csv"


def test_builtin_table_is_the_kit_sheet():
    sizes = read_tag_size_table(BUILTIN_TABLE)
    assert sorted(sizes) == list(range(48))
    assert all(sizes[t] == 0.4 for t in range(20, 28))
    assert all(sizes[t] == 0.25 for t in [*range(0, 20), *range(28, 48)])
    assert BUILTIN_TABLE.with_suffix(".pdf").is_file()


def test_table_formats_and_errors(tmp_path):
    metres = tmp_path / "m.csv"
    metres.write_text("﻿tag_id,size_m,family\n7,0.15,tag36h11\n8,0.3,tag25h9\n")
    assert read_tag_size_table(metres) == {7: 0.15}
    for text, match in [
        ("id,size_mm\n1,250\n", "tag_id"),
        ("tag_id,board\n1,300\n", "size column"),
        ("tag_id,size_mm\n1,abc\n", "line 2"),
        ("tag_id,size_mm\n1,250\n1,400\n", "twice"),
        ("tag_id,size_mm,family\n1,250,tag25h9\n", "no tag36h11 rows"),
    ]:
        bad = tmp_path / "bad.csv"
        bad.write_text(text)
        with pytest.raises(TagSizeTableError, match=match):
            read_tag_size_table(bad)


def test_device_choice_round_trips_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("VINE360_CONFIG_DIR", str(tmp_path / "config"))
    assert default_table_path() == BUILTIN_TABLE
    mine = tmp_path / "mine.csv"
    mine.write_text("tag_id,size_mm\n20,410\n")
    set_default_table_path(mine)
    assert default_table_path() == mine.resolve()
    mine.unlink()
    assert default_table_path() == BUILTIN_TABLE  # a vanished choice never breaks import
    set_default_table_path(None)
    assert default_table_path() == BUILTIN_TABLE


@pytest.fixture
def conn(tmp_path):
    root = tmp_path / "p"
    create_project(root, "sizes", CaptureMode.THREE_SIXTY)
    c = open_index_db(root)
    yield c
    c.close()


def test_import_sizes_points_from_the_table(conn):
    sizes = read_tag_size_table(BUILTIN_TABLE)
    set_id, warnings = import_control_points(conn, GCP_FIXTURE, epsg=7854, tag_size_m=0.3, tag_sizes=sizes)
    points = {p.name: p for p in load_control_points(conn, set_id)}
    assert points["020"].tag_size_m == 0.4 and points["027"].tag_size_m == 0.4
    assert points["000"].tag_size_m == 0.25 and points["005"].tag_size_m == 0.25
    assert points["PRS778941614462"].tag_size_m is None  # untagged
    assert not any("tag-size table" in w for w in warnings)


def test_unlisted_ids_fall_back_and_are_reported(conn):
    set_id, warnings = import_control_points(conn, GCP_FIXTURE, epsg=7854, tag_size_m=0.3, tag_sizes={20: 0.4})
    points = {p.name: p for p in load_control_points(conn, set_id)}
    assert points["020"].tag_size_m == 0.4 and points["021"].tag_size_m == 0.3
    assert any("not in the tag-size table, given the default 0.3 m" in w and "021" in w for w in warnings)


def test_apply_tag_sizes_to_an_existing_set(conn):
    set_id, _ = import_control_points(conn, GCP_FIXTURE, epsg=7854, tag_size_m=0.4)
    update_control_point(conn, set_id, "001", tag_size_m=0.9)  # a hand edit is replaced too
    changed, unlisted = apply_tag_sizes(conn, set_id, read_tag_size_table(BUILTIN_TABLE))
    points = {p.name: p for p in load_control_points(conn, set_id)}
    assert points["001"].tag_size_m == 0.25 and points["020"].tag_size_m == 0.4
    assert "001" in changed and "020" not in changed and unlisted == []
    assert apply_tag_sizes(conn, set_id, {20: 0.4})[1]  # ids not in a table are reported, left alone
    assert {p.name: p.tag_size_m for p in load_control_points(conn, set_id)}["001"] == 0.25
