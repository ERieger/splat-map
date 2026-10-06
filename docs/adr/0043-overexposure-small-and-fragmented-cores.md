# 0043. Overexposure: keep small and fragmented blown areas

## Status

Accepted. Amends ADR 0038 (core size) and builds on ADR 0042 (halo flood).

## Context

After ADR 0042, training on FMC-Tunnel1 still showed floaters at the far end of the tunnel and on
the walls around it. Measured on 500 rebuilt views:
- 74 views had clipped pixels but an *empty* overexposure layer.
- In others, the clipped area was only partly masked.

Two causes:
- **Small blown areas were dropped.** The distant tunnel end covers ~0.05–0.1% of a view, and a
  fluorescent fitting is similar. Both fell under ADR 0038's 0.2% minimum, which was meant to
  skip glints. The halo flood only starts from a surviving core, so the glow they cast on the
  walls was never looked at either.
- **Fragmented blown areas were dropped piece by piece.** Sky through leaves, a dappled sunlit
  path, or an exit seen past people or a railing is clipped in many small pieces. Each piece fell
  under the minimum on its own.

## Decision

In `classify_overexposed` (method version 3):
- `min_core_fraction` drops from 0.002 to **0.0002**, about 14×14 px at 1024². Real glints stay
  well under it.
- **Clipped pixels within 4 px of each other are grouped** before measuring. A group's size is its
  clipped-pixel count, not its dilated area.
- **A group must be at least 30% clipped**. Real blown areas measured 0.35–0.87, with only a few
  leafy-sky groups at 0.22–0.31. Clipped specks scattered through sunlit texture form large but
  sparse groups and stay excluded, which a synthetic speckled-hillside test enforces.

## Consequences

On 500 FMC-Tunnel1 views, compared with method version 2 (as rebuilt):
- **Distant tunnel ends and light fittings are masked**, and the glow around them now gets
  flooded too.
- **Unmasked near-clipped pixels (`min(RGB) >= 240`) per view:**
  - 90th percentile: 1.1% → 0.5%;
  - 99th percentile: 3.6% → 1.7%;
  - worst view: 6.2% → 2.2%.
- **Views with any overexposure mask:** 224 → 303. Views with clipped pixels but an empty layer:
  74 → 5.
- **The biggest growth is in looking-up views of a blown cloudy sky** (e.g. 3.6% → 15.8%). That
  sky is clipped and carries no information.
- **Largest mask in the sample:** 20%. About 0.18 s per 1024² view.

Limits:
- This change doesn't address walls that are bright but not near any clipped area. Those still
  have texture, and the layer deliberately keeps them.
- If such walls still produce floaters, the cause is likely exposure or the trainer, not a
  missing mask.
- Existing layers change only when rebuilt.
