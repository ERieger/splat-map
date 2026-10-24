"""Carrying a georeference solution's transform into files (docs/adr/0047):
a pycolmap reconstruction, a trained Gaussian-splat .ply, and the
georef.json / georef.prj sidecars.

Non-destructive by rule (ADR 0047 addendum): nothing here ever writes into
an SfM run's own model directory, and transform_splat_ply never overwrites
its input.

**pycolmap names verified against the installed pycolmap 4.2.1**:
`Sim3d(scale, Rotation3d, translation)` (overload 2 of Sim3d.__init__),
`Rotation3d(matrix_3x3)`, and `Reconstruction.transform(new_from_old_world:
Sim3d)` ("Apply the 3D similarity transformation to all images and
points"); `Sim3d * x` == scale * R @ x + t was checked by hand on that
install, and tests/test_georef_reconstruction.py pins the camera-centre and
point behaviour on a pycolmap.synthesize_dataset reconstruction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vine360.georef.similarity import Similarity


class GeorefApplyError(Exception):
    pass


# -- pycolmap ----------------------------------------------------------------


def sim3_from_similarity(similarity: Similarity):
    import pycolmap

    return pycolmap.Sim3d(
        float(similarity.scale), pycolmap.Rotation3d(np.asarray(similarity.rotation, dtype=np.float64)),
        np.asarray(similarity.translation, dtype=np.float64),
    )


def transform_reconstruction(reconstruction, similarity: Similarity):
    """Applies model -> local to every image pose and 3D point, in place on
    the (in-memory) reconstruction given -- callers pass one they loaded or
    built, never write it back over a run's own model -- and returns it."""
    reconstruction.transform(sim3_from_similarity(similarity))
    return reconstruction


# -- Gaussian-splat PLY ------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
CHUNK_VERTICES = 1 << 20


def _read_ply_header(f) -> tuple[bytes, int, np.dtype, int]:
    """(header bytes, vertex count, vertex dtype, offset) -- binary little
    endian, a single `vertex` element of scalar properties: the layout of
    the reference 3DGS implementation and Postshot's .ply export."""
    header = b""
    while True:
        line = f.readline()
        if not line:
            raise GeorefApplyError("not a PLY file (no end_header)")
        header += line
        if line.strip() == b"end_header":
            break
    lines = header.decode("ascii", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise GeorefApplyError("not a PLY file")
    fmt = next((l.split()[1] for l in lines if l.startswith("format ")), None)
    if fmt != "binary_little_endian":
        raise GeorefApplyError(f"only binary_little_endian PLY is supported, got {fmt!r}")
    elements = [l.split() for l in lines if l.startswith("element ")]
    if [e[1] for e in elements] != ["vertex"]:
        raise GeorefApplyError(f"expected a single 'vertex' element, got {[e[1] for e in elements]}")
    count = int(elements[0][2])
    fields = []
    for l in lines:
        parts = l.split()
        if parts[:1] == ["property"]:
            if parts[1] == "list":
                raise GeorefApplyError("list properties aren't supported in a splat PLY")
            if parts[1] not in _PLY_TYPES:
                raise GeorefApplyError(f"unknown PLY property type {parts[1]!r}")
            fields.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
    return header, count, np.dtype(fields), len(header)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b, wxyz; a (4,), b (N, 4)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def rotation_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    from vine360.projection.pose import rotation_matrix_to_quaternion_wxyz

    return np.asarray(rotation_matrix_to_quaternion_wxyz(np.asarray(R, dtype=np.float64)), dtype=np.float64)


def default_splat_output(in_path: Path, solution_id: str) -> Path:
    in_path = Path(in_path)
    return in_path.with_name(f"{in_path.stem}.georef-{solution_id[:8]}.ply")


def transform_splat_ply(in_path: Path, out_path: Path, similarity: Similarity, *, progress_callback=None) -> int:
    """Writes a new .ply with every Gaussian moved into the local metric
    frame; returns the number of Gaussians. Never touches in_path (refuses
    out_path == in_path).

    - positions x, y, z: s * R @ x + t
    - normals nx, ny, nz (if present): R @ n
    - orientations rot_0..3, **wxyz** (rot_0 = w -- the reference 3DGS
      implementation's and Postshot's convention): q' = q_R ⊗ q, magnitude
      unchanged (they're stored unnormalized)
    - scale_0..2 are log-scales: += ln(s)
    - opacity and spherical harmonics are copied unchanged. Degree-0 colour
      (f_dc_*) is rotation-invariant; **higher-degree SH (f_rest_*) are not
      rotated**, so view-dependent colour is slightly off after a large
      rotation (docs/adr/0047).

    Streams CHUNK_VERTICES at a time, so a multi-GB splat never sits in
    memory whole. Unverified against a real Postshot .ply (none available
    here) -- built to the reference layout and tested on a synthetic one.
    Only meaningful for a splat trained on an export *without* a
    georeference (the model frame); one trained on a georeferenced export is
    already local."""
    notify = progress_callback or (lambda *a: None)
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.resolve() == in_path.resolve():
        raise GeorefApplyError("refusing to transform a .ply in place -- choose a different output file")
    s = float(similarity.scale)
    R = np.asarray(similarity.rotation, dtype=np.float64)
    t = np.asarray(similarity.translation, dtype=np.float64)
    q_r = rotation_to_quaternion_wxyz(R)
    log_s = math.log(s)

    with in_path.open("rb") as src:
        header, count, dtype, _offset = _read_ply_header(src)
        names = set(dtype.names)
        if not {"x", "y", "z"} <= names:
            raise GeorefApplyError("PLY has no x/y/z vertex properties")
        rot = [f"rot_{i}" for i in range(4)]
        scales = [n for n in (f"scale_{i}" for i in range(3)) if n in names]
        has_rot = set(rot) <= names
        has_normals = {"nx", "ny", "nz"} <= names
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".partial")
        done = 0
        with tmp.open("wb") as dst:
            dst.write(header)
            while done < count:
                n = min(CHUNK_VERTICES, count - done)
                raw = src.read(n * dtype.itemsize)
                if len(raw) != n * dtype.itemsize:
                    raise GeorefApplyError(f"PLY ends early: {done + len(raw) // dtype.itemsize}/{count} vertices")
                block = np.frombuffer(raw, dtype=dtype).copy()
                xyz = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64)
                xyz = s * xyz @ R.T + t
                block["x"], block["y"], block["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                if has_normals:
                    normals = np.stack([block["nx"], block["ny"], block["nz"]], axis=1).astype(np.float64) @ R.T
                    block["nx"], block["ny"], block["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
                if has_rot:
                    q = _quat_multiply(q_r, np.stack([block[k] for k in rot], axis=1).astype(np.float64))
                    for i, k in enumerate(rot):
                        block[k] = q[:, i]
                for k in scales:
                    block[k] = block[k].astype(np.float64) + log_s
                dst.write(block.tobytes())
                done += n
                notify(f"Georeferencing splat: {done}/{count} Gaussians", done, count)
            trailing = src.read()
            dst.write(trailing)  # nothing, for a well-formed file -- kept rather than silently dropped
    tmp.replace(out_path)
    return count


# -- sidecars ----------------------------------------------------------------

# WKT1 (ESRI .prj style) for the projected CRSs v1 accepts. Parameters from
# the EPSG Geodetic Parameter Dataset (epsg.org; e.g. EPSG:7854 "GDA2020 /
# MGA zone 54": Transverse Mercator, lat0 0, central meridian 6*zone - 183,
# k0 0.9996, FE 500000, FN 10000000, GRS 1980).
_DATUMS = {
    "GDA2020": ("Geocentric_Datum_of_Australia_2020", 7852),
    "GDA94": ("Geocentric_Datum_of_Australia_1994", 28352),
}


def _mga_wkt(datum: str, zone: int, epsg: int) -> str:
    datum_name, _ = _DATUMS[datum]
    return (
        f'PROJCS["{datum} / MGA zone {zone}",GEOGCS["{datum}",DATUM["{datum_name}",'
        'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
        f'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",{6 * zone - 183}],'
        'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
        f'PARAMETER["false_northing",10000000],UNIT["metre",1],AUTHORITY["EPSG","{epsg}"]]'
    )


CRS_WKT = {
    base + (zone - 52): _mga_wkt(datum, zone, base + (zone - 52))
    for datum, (_, base) in _DATUMS.items()
    for zone in range(52, 57)
}


def crs_name(epsg: int) -> str | None:
    wkt = CRS_WKT.get(int(epsg))
    return wkt.split('"')[1] if wkt else None


def georef_metadata(solution: dict, cp_set) -> dict:
    """The georef.json content for a stored solution row (as returned by
    vine360.georef.solve.load_solution) and its control-point set."""
    return {
        "epsg": cp_set.epsg,
        "crs_name": crs_name(cp_set.epsg),
        "height_datum": cp_set.height_datum,
        "origin": list(cp_set.origin),
        "coordinates": "local = CRS - origin (E, N, H) in metres; add origin back for true CRS values",
        "model_to_local": solution["transform"],
        "control_point_set": cp_set.set_id,
        "control_point_file": cp_set.source_path,
        "run_id": solution["run_id"],
        "solution_id": solution["solution_id"],
        "created_at": solution["created_at"],
        "report": solution["report"],
    }


def write_georef_sidecars(directory: Path, metadata: dict) -> list[str]:
    """Writes georef.json, and georef.prj when the EPSG is one CRS_WKT
    knows; returns warnings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "georef.json").write_text(json.dumps(metadata, indent=2) + "\n")
    wkt = CRS_WKT.get(int(metadata["epsg"]))
    if wkt is None:
        return [f"no WKT known for EPSG:{metadata['epsg']} -- georef.prj not written"]
    (directory / "georef.prj").write_text(wkt)
    return []
