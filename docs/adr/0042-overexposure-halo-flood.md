# 0042. Overexposure bloom: follow the halo as it fades, not a fixed luma cut

## Status

Accepted. Amends the overexposure section of ADR 0038.

## Context

ADR 0038's overexposure layer masked the clipped core (`min(R,G,B) >= 250`, cores of at least
0.2% of the view) plus "bloom": luma ≥ 230 within 40 px of a core, in a component touching it.
On real FMC-Tunnel1 views this masked the blown tunnel exit and light fixtures, but left most of
the glow around them. The leftover glow is what training explains with floaters.

Measured on frame 90 front, the halo on the walls is a long, gentle ramp. It runs from ~235 at
the exit down to ~190 about 175 px away, on a wall whose normal luma is ~170. Brick mortar lines
break it into fragments. A 230 cut with a 40 px reach catches only its first ~25 px, and only the
fragments that touch the core. Frame 60's stored mask covered 1.8% of the view; the visible glow
is about three times that.

Approaches tried on real views and rejected:
- **Smoothing plus a radius scaled to core size.** Hardly any gain on tunnel exits. Under a blown
  cloudy sky the mask bled from the clouds into a sunlit hillside.
- **A lower luma floor with a per-pixel "brighter towards the core" gradient test.** Better on
  tunnels, but the hillside's texture passes the test about half the time.
- **An adaptive floor from the luma in a ring around the core.** In sunlit stair views the ring
  includes the dark tunnel interior, so it didn't raise the floor where it was needed.

## Decision

`classify_overexposed` (`masking/overexposure.py`, method version 2) now builds the mask in
stages:
1. **Core:** unchanged.
2. **Glare:** the old ADR 0038 bloom rule, kept as-is (luma ≥ 230 within 40 px of a core,
   connected to it). This keeps the washed-out rim of the opening and the barely visible scene
   through it masked, so the new mask covers everything the old one did.
3. **Halo:** a flood outward from core + glare, run on a quarter-resolution, Gaussian-smoothed
   luma image. A pixel joins the flood only if it is all of these:
   - at least `bloom_threshold` bright (now 185);
   - within `bloom_radius_px` of the clipped core (now 160);
   - smooth: local luma std (9 px window) ≤ `bloom_max_texture` (12). In real views the halo's
     median is ~9 and a sunlit hillside's is ~17. Glare washes detail out.
   - at least 0.1 darker than the brightest flooded neighbour. Bloom fades away from the light,
     so the flood stops on a plateau.

   Holes enclosed by the flood are filled.
4. **Dilation:** by `dilation_px`, using a disc instead of the old square (which drew blocky
   rectangles).

The falloff requirement was swept on real and synthetic views:
- **0.25:** too strict. It lost frame 60's left-wall glow (4.5% → 3.6%).
- **≤ 0:** a perfectly flat bright wall beside an opening floods the full 160 px.
- **0.1:** keeps most of the halo, and limits a flat wall to the ~23 px smoothing ramp.

## Consequences

- On FMC-Tunnel1, tunnel exits now lose their halo:
  - frame 60 front: 1.8% → 4.7%;
  - frame 90 front: 7.2% → 13%;
  - frame 45 back: 4.5% → 12%.

  White, green and yellow brick walls away from exits, and sunlit hillsides under a blown sky,
  stay unmasked.
- On 400 random views, the share of views with any overexposure is unchanged. Coverage has a 99th
  percentile of 10.5% and a maximum of 11%.
- On 150 views, nothing the old mask covered is uncovered, except within 6 px of the new mask's
  edge (disc vs square dilation).
- ~0.1–0.2 s per 1024² view, about the same as before.
- New layer parameter `bloom_max_texture`. Rows built before this change lack it, and
  `compute_layer` falls back to the default.
- The GUI bloom spin box defaults to 160 px, and its tooltip explains that the halo only spreads
  over smooth, fading areas.
- Existing projects keep their old masks until they are rebuilt. `method_version` "2" on a layer
  row identifies masks built by this algorithm.
- Still a heuristic, tuned by eye on one capture. There is no ground truth for "bloom".
  - A smooth, evenly lit surface that brightens towards an opening (real daylight spill, not lens
    bloom) is masked within the reach. The two can't be told apart from one image, and that
    surface is washed out anyway.
  - Tune by lowering `bloom_radius_px` in the Masks panel.
