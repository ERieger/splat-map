"""vine360.georef.apply: the splat PLY transform on a tiny synthetic
3DGS-layout PLY, hand-checked; sidecars; and pycolmap's Sim3d /
Reconstruction.transform on a real synthesized reconstruction
(docs/adr/0047)."""

import json
import math

import numpy as np
import pytest

from georef_synthetic import random_rotation
from vine360.georef import apply as georef_apply
from vine360.georef.apply import (
    CRS_WKT,
    GeorefApplyError,
    crs_name,
    default_splat_output,
    transform_splat_ply,
    write_georef_sidecars,
)
from vine360.georef.similarity import Similarity
from vine360.projection.pose import quaternion_wxyz_to_rotation_matrix

FIELDS = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(3)]
          + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])


def _write_ply(path, data):
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(data)}\n"
    header += "".join(f"property float {name}\n" for name in FIELDS) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4").tobytes())


def _read_ply(path):
    raw = path.read_bytes()
    body = raw[raw.index(b"end_header\n") + len(b"end_header\n"):]
    return np.frombuffer(body, dtype="<f4").reshape(-1, len(FIELDS)).astype(np.float64)


def test_splat_ply_positions_rotations_and_log_scales(tmp_path, monkeypatch):
    monkeypatch.setattr(georef_apply, "CHUNK_VERTICES", 3)  # exercise streaming across chunks
    rng = np.random.default_rng(0)
    n = 7
    data = rng.normal(size=(n, len(FIELDS)))
    data[:, 3:6] /= np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    src = tmp_path / "splat.ply"
    _write_ply(src, data)
    data = _read_ply(src)  # float32-rounded values, as the transform will see them
    sim = Similarity(0.42, random_rotation(rng), np.array([12.0, -3.0, 360.0]))
    out = default_splat_output(src, "geo-1234abcd")
    assert out.name == "splat.georef-geo-1234.ply"
    assert transform_splat_ply(src, out, sim) == n

    result = _read_ply(out)
    col = {name: i for i, name in enumerate(FIELDS)}
    xyz = data[:, :3]
    np.testing.assert_allclose(result[:, :3], 0.42 * xyz @ sim.rotation.T + sim.translation, atol=1e-4)
    np.testing.assert_allclose(result[:, 3:6], data[:, 3:6] @ sim.rotation.T, atol=1e-5)
    scales = [col[f"scale_{i}"] for i in range(3)]
    np.testing.assert_allclose(result[:, scales], data[:, scales] + math.log(0.42), atol=1e-5)
    unchanged = [col[k] for k in ("f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "f_rest_1", "f_rest_2", "opacity")]
    np.testing.assert_array_equal(result[:, unchanged], data[:, unchanged])
    rot = [col[f"rot_{i}"] for i in range(4)]
    for before, after in zip(data[:, rot], result[:, rot]):
        assert np.linalg.norm(after) == pytest.approx(np.linalg.norm(before), rel=1e-5)  # magnitude kept
        expected = sim.rotation @ quaternion_wxyz_to_rotation_matrix(before / np.linalg.norm(before))
        np.testing.assert_allclose(quaternion_wxyz_to_rotation_matrix(after / np.linalg.norm(after)), expected, atol=1e-5)
    assert src.read_bytes()[:3] == b"ply" and not list(tmp_path.glob("*.partial"))


def test_splat_ply_is_never_transformed_in_place(tmp_path):
    src = tmp_path / "splat.ply"
    _write_ply(src, np.zeros((2, len(FIELDS))))
    before = src.read_bytes()
    with pytest.raises(GeorefApplyError, match="in place"):
        transform_splat_ply(src, src, Similarity(1.0, np.eye(3), np.zeros(3)))
    assert src.read_bytes() == before


def test_rejects_ascii_ply(tmp_path):
    src = tmp_path / "a.ply"
    src.write_text("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n")
    with pytest.raises(GeorefApplyError, match="binary_little_endian"):
        transform_splat_ply(src, tmp_path / "b.ply", Similarity(1.0, np.eye(3), np.zeros(3)))


def test_sidecars_and_wkt(tmp_path):
    assert crs_name(7854) == "GDA2020 / MGA zone 54" and crs_name(28356) == "GDA94 / MGA zone 56"
    assert 'PARAMETER["central_meridian",141]' in CRS_WKT[7854]
    assert set(CRS_WKT) == {7852, 7853, 7854, 7855, 7856, 28352, 28353, 28354, 28355, 28356}
    assert write_georef_sidecars(tmp_path, {"epsg": 7854, "origin": [1, 2, 0]}) == []
    assert json.loads((tmp_path / "georef.json").read_text())["epsg"] == 7854
    assert (tmp_path / "georef.prj").read_text().startswith('PROJCS["GDA2020 / MGA zone 54"')
    warnings = write_georef_sidecars(tmp_path / "other", {"epsg": 32754})
    assert warnings and not (tmp_path / "other" / "georef.prj").exists()


def test_reconstruction_transform_moves_cameras_and_points():
    pycolmap = pytest.importorskip("pycolmap")
    from vine360.georef.apply import sim3_from_similarity, transform_reconstruction

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig, options.num_points3D = 1, 1, 5, 50
    reconstruction = pycolmap.synthesize_dataset(options)
    centers = {i: np.array(im.projection_center()) for i, im in reconstruction.images.items()}
    points = {i: np.array(p.xyz) for i, p in reconstruction.points3D.items()}
    rng = np.random.default_rng(1)
    sim = Similarity(2.5, random_rotation(rng), np.array([1.0, 2.0, 3.0]))
    # The hand-computed point: Sim3d * x == s * R @ x + t.
    np.testing.assert_allclose(sim3_from_similarity(sim) * np.array([1.0, 0.0, 0.0]), sim.apply(np.array([[1.0, 0.0, 0.0]]))[0])
    transform_reconstruction(reconstruction, sim)
    for i, image in reconstruction.images.items():
        np.testing.assert_allclose(image.projection_center(), sim.apply(centers[i][None])[0], atol=1e-9)
    for i, point in reconstruction.points3D.items():
        np.testing.assert_allclose(point.xyz, sim.apply(points[i][None])[0], atol=1e-9)
