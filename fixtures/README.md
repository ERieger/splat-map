Reserved for a small, redistributable reference capture (see handover doc,
section 12: "What small, redistributable reference dataset may be
committed or linked for automated end-to-end testing?"). Nothing is
committed here yet -- `tests/test_ingest_integration.py` currently
generates a synthetic clip with `ffmpeg`'s `testsrc` source instead.

## gcp/ -- real ground-control survey files (docs/adr/0047)

Copied byte-for-byte from each capture's `gcp/` folder on 2026-10-25 (`-text` in
.gitattributes keeps their encodings and CRLFs intact):

| File | From | Shape |
|---|---|---|
| `ESTO1_GCPs_23Oct26.csv` | `2026-10-23_EstoWines_Capture2/gcp/` | UTF-8, no header, trailing comma; first row is the GNSS reference station |
| `esto_cap1_gcp_utf16.csv` | `2026-09-11_EstoWines_Capture1/gcp/Esto Wines Cap1 GCP.csv` | UTF-16 + BOM, header row |
| `esto_cap1_precision_utf16.csv` | `…/Esto Wines Cap1 with precision.csv` | as above plus H/V GNSS precision columns |
| `flinders_bridge_mga2020.csv` | `2026-09-30_Flinders-Bridge/gcp/Grant Control MGA2020.csv` | UTF-8, no header, `GCP000`-style names |

Capture 1 and Capture 2 reuse point names at different locations -- never mix them.

## apriltag/ -- reference tag36h11 images

`tag36_11_000NN.png` are the official AprilRobotics `apriltag-imgs` renders (commit
f3fd9a7, BSD-2-Clause, (c) 2013-2016 The Regents of The University of Michigan): 10×10 px,
one pixel per cell, white outer border. Scale them up with nearest-neighbour before
detecting.
