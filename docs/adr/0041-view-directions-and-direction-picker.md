# 0041. View directions beyond the cubemap, and a graphical direction picker

## Status

Accepted.

## Context

Projection could only render the six cubemap faces (front/right/back/left/up/down), chosen with
six checkboxes. In a tight vine-row "tunnel" this works badly. The level faces see the walls but
little floor, the straight-down face is a strongly foreshortened, low-texture view of the ground,
and neighbouring level faces only touch at their edges (90° apart with a 90° field of view), so
SfM gets little overlap between them. A view tilted down along a diagonal sees a row wall *and*
the floor in one image, which links the two surfaces in matching.

## Decision

**A fixed catalog of 26 named directions** (`vine360/projection/cubemap.py`):
- yaw every 45° on three rings: the horizon (pitch 0), a lower ring (pitch −tilt) and an upper
  ring (pitch +tilt);
- plus the two poles.

Names are `front`, `front-right`, `right`, …; the tilted rings append `-down`/`-up`
(`front-left-down`); the poles stay `up`/`down`. Every view has zero roll, so the horizon stays
level.

- **The six legacy names keep exactly their old rotations.** Existing view ids
  (`<frame_id>:front`), queued jobs and every downstream consumer are unaffected. SfM pose
  composition, the exports and masks all read each view's stored `fixed_rotation` and name, and
  never assumed 90° steps, so they needed no logic changes.
- **The ring tilt is a parameter** (`ring_tilt_degrees`, default 45°, range 10–80°), not baked
  into the name: `front-down` is "the lower ring" at whatever tilt it was generated with.
  - At 45° with a 90° field of view, a lower-ring view spans exactly horizon-to-nadir.
  - A shallower tilt keeps more wall; a steeper one more floor.
  - The real angle is stored in each view's `fixed_rotation` and recorded as a worker parameter,
    so it appears in the Activity log. The GUI recovers it from `fixed_rotation` to preselect an
    existing frame set's settings.
  - Queued jobs from before this change carry no tilt; they default to 45°, which is irrelevant
    to them because they only name level or polar faces.
- **A fixed catalog, not free yaw/pitch entry.** Names stay meaningful in file names, view ids,
  the Masks panel's face browser and exports (`<frame>__front-left-down.png`). 45° steps give 50%
  overlap between neighbouring horizon views at 90° FOV. If a free-form direction is ever needed,
  it can be added as another catalog entry kind.
- **`projection_id` is now `"directions"`** (it was `"six-face"`). Nothing reads it; existing rows
  keep their old value.

**A graphical picker** (`vine360/gui/direction_picker.py`) replaces the checkboxes:
- every catalog direction is a clickable marker drawn on the current frame's own panorama (its
  thumbnail, falling back to the frame);
- each selected direction's footprint (what that view will see at the current field of view and
  tilt) is outlined on the panorama. Outlines come from `cubemap.footprint_outline`, which is split
  at the ±180° seam;
- hovering a marker renders a small live preview of that view;
- presets cover the common selections: cardinal, cardinal + diagonals, vine-row tunnel,
  horizon + lower ring, and full sphere;
- a summary line shows views per frame × frames = total images.

We picked drawing on the real panorama over an abstract compass dial: it shows what each view
will actually contain, which is the information needed to pick directions for a given row.

## Consequences

- Up to 26 views per frame. Masking and SfM cost scale with image count. The picker's summary
  line warns above 8 views per frame. The default selection (four cardinal faces) is unchanged.
- All views of one frame still share one optical centre (zero baseline between them), the same as
  before.
- Thumbnail ordering in the Projection preview and the Masks face combo follows catalog order
  (`direction_sort_key`), not alphabetical `view_id`.
- The Projection panel is taller. At about 800 px of panel height, the per-frame preview gallery
  below the picker gets little room.
