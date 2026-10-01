# 0028. RealityScan export mode; nested `<capture>/<format>/` export layout

## Status

Accepted.

## Context

The user wanted a third export flavor for RealityScan (Epic Games), for it to run its own pose
estimation on vine360's already-projected images — the same "let the external tool align its own
photos" shape as the existing Postshot images-only mode, just with RealityScan's own conventions.
They also wanted the export directory layout restructured: every export still gets its own
per-capture folder (ADR 0026), but now with a further subfolder naming the output *format*, and
wanted already-exported folders on disk migrated into the new layout automatically.

Real research grounded the RealityScan side (its own published docs, not guessed):

- [Align Images - RealityScan Help](https://rshelp.capturingreality.com/en-US/tutorials/quickstart_2.htm)
- [RealityScan 2.2 Adds AMD GPU Support, 360 Camera Workflow Improvements](https://www.motionmedia.com/mm-blog/realityscan-2-2-amd-gpu-support-360-camera-workflow)
- [Mask Images - RealityScan Help](https://rshelp.capturingreality.com/en-US/tools/mask.htm)
- [Image Layers - RealityScan Help](https://rshelp.capturingreality.com/en-US/tools/imglayers.htm)

Key findings: RealityScan's documented, recommended 360-footage workflow is to convert
equirectangular images into perspective cube-face images before alignment — exactly what vine360's
Projection stage already produces, so no raw-equirectangular export mode was needed (confirmed with
the user, who declined a separate "equi" format). RealityScan documents two mask-association
conventions: a flat `<image-filename>.mask.png` sitting next to its color image in the same folder,
or a dedicated `layers/.mask/` subfolder. It can also optionally take XMP sidecar files with camera
calibration to speed up alignment.

## Decision

**Two new/changed things, both confirmed with the user:**

1. **`export_for_realityscan`** (`vine360/export/postshot.py`), a third sibling of `export_for_
   postshot`/`export_frames_and_masks_for_postshot` in the same module (kept in the "currently
   established framework" rather than a new module). Same prerequisite, error messages, and
   `source_id` filter as `export_frames_and_masks_for_postshot`; the only difference is the mask
   convention. **Superseded shortly after landing**: initially used RealityScan's flat, adjacent
   convention (`<flattened-image-name>.mask.png` next to its color image in the single `images/`
   output folder) over the `layers/.mask/` subfolder alternative — then switched to the
   **`layers/.mask/` subfolder** convention instead, once the user reported wanting to also import
   the same exported images into Postshot separately, and wanted to be able to select "just the
   images" without RealityScan's `.mask.png` files cluttering that folder listing. Confirmed real
   (not guessed) before making the switch: RealityScan auto-detects and associates masks placed in
   `images/layers/.mask/<same-filename>.png` by filename, the same way it does the flat-adjacent
   form — same source as originally cited below. **XMP sidecar files are skipped for
   v1** — the user chose this over a best-effort writer, since the exact schema isn't confirmed
   against a real RealityScan install (this environment has no RealityScan available, same
   honesty-discipline caveat ADR 0018 already applies to the Postshot export).

2. **Nested export directory layout**: `exports/<capture>/<format>/` instead of the flat
   `exports/<capture>/` ADR 0026 introduced. `<format>` is `colmap` (poses mode), `postshot`
   (frames_masks mode, nested for consistency even though its content is unchanged), or
   `realityscan` (the new mode). The all-sources default capture-folder is renamed from
   `"postshot"` to `"all"`, since `"postshot"` is now a *format* tag and reusing it as the
   all-sources capture name too would collide in meaning. Poses-mode capture-folder naming gets
   one small enhancement: if the selected SfM run's own stored config shows it was scoped to one
   source (the native-equirectangular Pose-estimation engine), the capture folder defaults to that
   source's name instead of `"all"` — a nicer default path only, poses mode still has no per-source
   *content* filter (ADR 0026's documented, deliberate gap still stands).

**Legacy-folder migration**, confirmed with the user (auto-migrate rather than leave old exports
alone): `_migrate_legacy_export_layout(capture_dir)`, called at the top of all three export
functions before they touch their own nested `output_dir`. Looks for the old flat
`capture_dir/{images,masks,sparse}/` shape; infers which format it was from its own contents
(`sparse/` present → `colmap`; images/masks with no `sparse/` → `postshot` — the only two modes
that ever wrote that flat shape, so `realityscan` is never inferred here) and moves it into
`capture_dir/<inferred-format>/`. Conservative, matching this project's established "never guess
under ambiguity" precedent (`vine360.sfm.repair_selected_model`, this session's own
`QueueManager._resolve_prerequisites`): a no-op if there's no legacy folder, if the inferred
nested target already exists, or if the flat folder doesn't match either recognized shape.

## Consequences

- `ExportPanel`'s mode combo, default output path, worker dispatch, and `QueueManager`'s
  `STAGE_EXPORT` dispatch all gained a third `"realityscan"` branch. The queue's prerequisite
  resolution needed **no changes** — it already only special-cased `"poses"`, so `"realityscan"`
  automatically gets the same correct, source-scoped Projection-prerequisite handling
  `"frames_masks"` already had.
- A project with exports made before this change gets them reorganized into the new layout
  automatically the next time it's exported to (not proactively on project open — the migration is
  scoped to the specific capture folder an export function is about to write into, not a project-
  wide sweep). A one-off proactive sweep helper was briefly added and wired into `ExportPanel.
  on_shown`, then deliberately removed at the user's request: with only two real projects and no
  intent to run older versions of the software that would produce more legacy-shaped folders, a
  permanent app-level sweep wasn't worth keeping around. Both real projects (`/mnt/e/TEST` and
  `/mnt/e/11-9-26_EstoWines_Capture1/A1_InstaLow`) were migrated once, directly, using the sweep
  before it was removed -- their `exports/` trees already reflect the new layout.
- `export_for_realityscan` carries the same unverified-against-a-real-install caveat as
  `export_for_postshot` (ADR 0018) — test importing a small export into RealityScan before relying
  on this for a real project.
