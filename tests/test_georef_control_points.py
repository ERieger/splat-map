"""vine360.georef.control_points against the real survey files
(fixtures/gcp/, copied unmodified from each capture's gcp/ folder) --
docs/adr/0047."""

from pathlib import Path

import pytest

from vine360.config import CaptureMode
from vine360.georef.control_points import (
    USE_CHECK,
    USE_CONTROL,
    USE_IGNORE,
    ControlPointError,
    default_origin,
    derive_tag_id,
    get_control_point_set,
    import_control_points,
    list_control_point_sets,
    load_control_points,
    read_control_csv,
    update_control_point,
)
from vine360.project import create_project, open_index_db

GCP = Path(__file__).resolve().parent.parent / "fixtures" / "gcp"
CAPTURE2 = GCP / "ESTO1_GCPs_23Oct26.csv"


def test_capture2_file_headerless_trailing_comma():
    parsed = read_control_csv(CAPTURE2)
    assert not parsed.has_header and parsed.encoding == "utf-8"
    assert len(parsed.points) == 15
    names = [p.name for p in parsed.points]
    assert names[0] == "PRS778941614462" and "000" in names and "020" in names
    by_name = {p.name: p for p in parsed.points}
    assert by_name["000"].tag_id == 0 and by_name["020"].tag_id == 20 and by_name["005"].tag_id == 5
    assert by_name["020"].east == pytest.approx(298980.109) and by_name["020"].north == pytest.approx(6126217.946)
    assert by_name["020"].height == pytest.approx(362.220)
    assert all(p.sigma_h is None and p.sigma_v is None for p in parsed.points)


def test_reference_station_is_flagged_ignore_with_a_reason():
    parsed = read_control_csv(CAPTURE2)
    base = parsed.points[0]
    assert base.tag_id is None  # far too large for tag36h11
    assert base.use == USE_IGNORE and "10.4 km" in base.note
    assert any("PRS778941614462" in w for w in parsed.warnings)
    assert all(p.use == USE_CONTROL for p in parsed.points[1:])


def test_default_origin_floors_to_100m_excluding_ignored_points():
    assert default_origin(read_control_csv(CAPTURE2).points) == (298900.0, 6126100.0, 0.0)


def test_utf16_header_file():
    parsed = read_control_csv(GCP / "esto_cap1_gcp_utf16.csv")
    assert parsed.has_header and parsed.encoding == "utf-16"
    assert [p.name for p in parsed.points][:3] == ["000", "001", "002"] and len(parsed.points) == 14
    assert parsed.points[0].east == pytest.approx(299021.978)
    assert all(p.use == USE_CONTROL for p in parsed.points)


def test_utf16_header_file_with_precision_columns():
    parsed = read_control_csv(GCP / "esto_cap1_precision_utf16.csv")
    first = parsed.points[0]
    assert (first.sigma_h, first.sigma_v) == (pytest.approx(0.017), pytest.approx(0.034))
    assert parsed.points[-1].name == "027" and parsed.points[-1].sigma_v == pytest.approx(0.035)


def test_prefixed_names_get_tag_ids():
    parsed = read_control_csv(GCP / "flinders_bridge_mga2020.csv")
    assert [p.name for p in parsed.points][:2] == ["GCP000", "GCP001"]
    assert [p.tag_id for p in parsed.points] == list(range(12))


def test_header_detection_on_a_utf8_variant(tmp_path):
    path = tmp_path / "with_header.csv"
    path.write_text("name,easting,northing,height\n020,1.5,2.5,3.5\n021,4,5,6\n")
    parsed = read_control_csv(path)
    assert parsed.has_header and [p.name for p in parsed.points] == ["020", "021"]
    # A numeric first row is never mistaken for a header.
    path.write_text("020,1.5,2.5,3.5\n")
    assert not read_control_csv(path).has_header


def test_derive_tag_id():
    assert derive_tag_id("020") == 20
    assert derive_tag_id("000") == 0
    assert derive_tag_id("GCP011") == 11
    assert derive_tag_id("PRS778941614462") is None
    assert derive_tag_id("587") is None  # tag36h11 ids are 0..586
    assert derive_tag_id("fence post") is None


def test_bad_rows_name_the_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("001,1,2,3,\n002,1,x,3,\n")
    with pytest.raises(ControlPointError, match="line 2"):
        read_control_csv(path)
    path.write_text("001,1,2,3,\n001,1,2,3,\n")
    with pytest.raises(ControlPointError, match="duplicate"):
        read_control_csv(path)


def test_import_list_and_update_in_the_index(tmp_path):
    create_project(tmp_path / "p", "p", CaptureMode.THREE_SIXTY)
    conn = open_index_db(tmp_path / "p")
    set_id, warnings = import_control_points(conn, CAPTURE2, epsg=7854, tag_size_m=0.4)
    assert warnings
    meta = get_control_point_set(conn, set_id)
    assert meta.epsg == 7854 and meta.height_datum == "unknown" and meta.tag_family == "tag36h11"
    assert meta.origin == (298900.0, 6126100.0, 0.0)
    points = load_control_points(conn, set_id)
    assert [p.name for p in points][:2] == ["PRS778941614462", "020"]  # file order kept
    assert points[1].tag_size_m == 0.4 and points[0].tag_size_m is None

    update_control_point(conn, set_id, "005", use=USE_CHECK, tag_size_m=0.25)
    update_control_point(conn, set_id, "027", tag_id=31)
    by_name = {p.name: p for p in load_control_points(conn, set_id)}
    assert by_name["005"].use == USE_CHECK and by_name["005"].tag_size_m == 0.25
    assert by_name["027"].tag_id == 31
    with pytest.raises(ControlPointError):
        update_control_point(conn, set_id, "005", use="maybe")

    second, _ = import_control_points(conn, CAPTURE2, epsg=7854)
    assert {s.set_id for s in list_control_point_sets(conn)} == {set_id, second}
    with pytest.raises(ControlPointError, match="EPSG"):
        import_control_points(conn, CAPTURE2, epsg=0)
