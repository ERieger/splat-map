"""Equirectangular-to-perspective projection (handover doc, section 4, step 3).

See pose.py for the camera-to-world pose convention and the
T_world_face = T_world_panorama . T_panorama_face composition, geometry.py
for the equirectangular/camera math, cubemap.py for the six-face 90-degree
preset, and render.py for the actual image resampling.
"""
