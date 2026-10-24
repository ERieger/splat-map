"""Georeferencing (M8, docs/adr/0047): surveyed ground control points marked
with AprilTags turn an SfM run's arbitrary model frame into a metric frame
local to a projected CRS (CRS minus a per-set origin).

Pure numpy here and in control_points/triangulate/similarity; the detector
(apriltag_adapter, the `georef` extra) and pycolmap (apply) are imported only
where they're used.
"""
