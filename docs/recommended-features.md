# Recommended missing features — Vine360 (review draft, 2026-10-08)

## Context
The user asked which features or functionality the app is missing. I compared the handover
spec (`Vineyard_360_3DGS_Software_Handover.docx`: P0/P1 table, §4 pipeline, §6 workflow,
§9 acceptance tests A01–A08) against `docs/status.md` (including "Blockers / known gaps"), the
ADRs up to 0045, and grep checks of `src/vine360`. Nothing below is implemented yet. Each item
would get its own ADR and its own task.

## Tier 1: biggest gaps for real vineyard results
1. **Rig-aware 360 SfM** (spec §4 "Estimate poses" and risk table). Views from one panorama are
   still mapped as independent images (grep found no rig config in `sfm/`). Recent pycolmap
   versions support rigs (`RigConfig` / `apply_rig_config`), so all faces of a frame could share
   one optical centre with fixed relative rotations. The conversion math already exists in
   `sfm/frame_poses.py` and `projection/pose.py`. This is likely the most effective fix for
   ground-level row tunnels that don't register.
2. **Per-face drift diagnostic** (spec says this is MVP). Measure how far each frame's
   independently registered faces move from a shared centre, and show it with the existing
   registration stats in the Pose panel. It is cheap to build and also measures whether item 1
   helps.
3. **Run the pipeline on real footage** (status "next task" #5). Nothing has run on the
   EstoWines clips yet. A short, explicitly approved run of one row segment would test the
   tunnel preset (ADR 0041) and the overexposure layers.

## Tier 2: spec P0/P1 items not yet built
4. **Near-duplicate and blurry frame filtering** (spec §4.2; still open). When the camera is
   stopped or moving slowly, frames repeat. Filter them by sharpness (variance of the Laplacian)
   and by inter-frame similarity at extraction time. Store the decisions in the frame set and
   show the rejected frames in the Frames panel.
5. **In-app mask brush correction** (spec P0 "edit", M3). Layers can be hand-edited outside the
   app (ADR 0038), but there's no brush in the app. Add a paint/erase "manual" layer on the
   review view. The existing layer merge and the `edited` flag already handle the downstream
   work.
6. **Provenance manifest in exports** (spec A06, §8). The Postshot and RealityScan exports
   should include a `manifest.json` with source checksums, frame-set config, projection
   directions, mask layer methods and versions, SfM options, engine versions and run IDs. Much
   of this is already in `activity_log` and the `sfm_runs` rows.
7. **Sparse model preview** (spec §6 step 7). Add a simple 3D point-cloud and camera-frustum
   view of the selected SfM model in the Pose panel. Basic matplotlib or a Qt3D/pyqtgraph
   scatter is enough to spot broken or split models before export.
8. **GPX georeferencing (M8)**. The A1 footage has `.gpx` sidecars. Interpolate a position for
   each frame and use it (a) as RealityScan camera priors without needing an SfM run, and (b) to
   compute a documented similarity transform for the SfM model.

## Tier 3: robustness and UX
9. **Path/disk preflight** (spec A07, §6 step 2). Before each stage, check that paths with
   spaces or non-ASCII characters work (or give a clear error), and that there's enough free
   disk for the estimated output size (`shutil.disk_usage`).
10. **Contact sheet before full extraction** (spec §6 step 3). Show a quick N-frame sample grid
    for the chosen interval before running the full extraction.
11. **Resume safety (A05)**. `ingest/frames.py` notes that crash-resume isn't implemented. Mark
    rows from an interrupted run as stale on open, and let the queue resume them.
12. **Automated GUI tests** (status "next task" #1). Turn the manual offscreen smoke flow into
    pytest tests.
13. **Mixed capture with conventional photos (M7)**. Import exists, but no conventional-camera
    group runs through SfM alongside the 360 views. This is lower priority because both real
    rigs are 360 (ADR 0010).

## Deliberately excluded
Training and viewer integration (ADR 0025 / 0009: out of scope by decision) and repeat-survey
comparison (P2).

## Suggested order
2 → 1 → 3 (drift measurement first, so the rig change has a baseline), then 6, 4, 5.

## Verification (per item when built)
Real-fixture pytest following repo conventions (`pycolmap.synthesize_dataset`,
`tests/synthetic_360.py`), an ADR, a `docs/status.md` update, and wrapping any new
long-running worker in `activity_log.logged_operation` with counted progress (ADR 0037).
