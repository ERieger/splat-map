# Status

## Completed work

**M0 -- Repository.** CLI scaffold, dependency probe (ffmpeg/ffprobe/
colmap/git/nvidia-smi, never raises on a missing tool), `Runner`
abstraction, pytest harness. Unchanged since the first pass; see the ADRs
below for what's since layered on top.

**M1 -- Ingest.** `vine360 project create/info`, `ingest add-source`
(ffprobe metadata, checksum, equirectangular confirmation gate),
`ingest remove-source` (added 2026-09-20 -- deletes the registered source
and its extracted frames, never the original file),
`ingest extract-frames` (deterministic ffmpeg extraction + thumbnails +
source-frame map; now clears a source's prior frames before re-extracting
-- see ADR 0011, this was a real "UNIQUE constraint failed" bug found via
use), `ingest manifest`. `ingest extract-frames` now also accepts
`--start-time`/`--end-time` (GUI: a "Limit to a time range" checkbox on
the Frames panel) to sample only a sub-range of a source instead of the
whole file -- see ADR 0031. It also accepts `--skip-thumbnails` (GUI: a
"Generate thumbnails" checkbox, on by default) to skip the per-frame
thumbnail pass -- nothing in the app currently reads
`frames/<source_id>/thumbs/` back, so this is a real time saver on a
large source with no loss of function today. The real end-to-end
integration test now actually runs (previously skipped for lack of ffmpeg) against
`static-ffmpeg`'s bundled real binaries.

**M2 -- Projection (`vine360/projection/`), fully implemented as library
code, not yet CLI-wired.**
- `pose.py`: camera-to-world `Pose`, `compose()` implementing
  `T_world_face = T_world_panorama . T_panorama_face`, and COLMAP
  qvec/tvec conversion.
- `geometry.py`: equirectangular <-> direction, perspective camera <->
  direction, the six-face rotation math.
- `cubemap.py`: the six-face 90-degree preset, polar faces excluded by
  default per the handover doc's own guidance.
- `render.py`: real numpy/Pillow bilinear resampling from an
  equirectangular image to a perspective face, including seam wraparound.
- Verified two ways: unit tests to 1e-9 precision (`test_projection_
  geometry.py`, `test_projection_pose.py`), and a visual sanity check
  (rendered a labeled synthetic panorama, inspected the front/right/up
  face outputs directly) that showed exactly the expected marker
  centering and polar radial-line convergence.
- A01 (round-trip projection within one pixel) passes against the actual
  transform; the image-based tests use a documented, wider pixel
  tolerance to account for marker-block quantization under bilinear
  resampling, not transform error (see the comment in
  `test_projection_render.py`).
- `generate_views_for_source` can now render frames in parallel across a
  capped number of worker processes (`max_workers`, GUI: a "Speed up with
  parallel processing" checkbox on the Projection panel, on by default) --
  the first use of `multiprocessing` anywhere in this codebase. Verified
  for real: 16.3s sequential vs. 9.0s with 4 workers on a 12-frame, 4K
  synthetic benchmark. See ADR 0032.

**M3 -- Masking (`vine360/masking/`), fully implemented, not yet
CLI-wired.**
- `semantics.py`: mask value conventions, small-component removal +
  dilation (deliberately reordered from the doc's literal step order --
  see ADR 0008), keep/exclude mask derivation, `keep_fraction` anomaly
  flagging, COLMAP mask path convention.
- `classical_sky.py`: a weights-free sky heuristic (brightness/blue-
  dominance/top-of-frame), explicitly a stand-in for SAM 3, not a
  substitute for it. No equivalent exists (or can reasonably exist) for
  person masks.
- `sam3_adapter.py`: the real backend against `transformers`' `Sam3Model`/
  `Sam3Processor`. Confirmed against a live call to the actual (gated,
  currently unauthenticated) `facebook/sam3` repo -- caught and fixed a
  real bug where `transformers` surfaces the gating failure as a plain
  `OSError`, not `huggingface_hub`'s `GatedRepoError`.

**M4 -- SfM (`vine360/sfm/colmap_adapter.py`), fully implemented, not yet
CLI-wired.**
- Uses `pycolmap` (COLMAP's own Python bindings) instead of a subprocess
  `colmap` binary -- see ADR 0007. `extract_and_match` (masked feature
  extraction + sequential matching) and `map_and_diagnose` (incremental
  mapping + the section-7 diagnostics) are separate, independently
  testable functions; `run_sfm` composes them for the real-files case.
- Real, unmocked tests: A04 ("no keypoints inside a zero-valued mask
  region") against actual COLMAP database output; a documented,
  *asserted* failure case showing that views rendered from a single
  panorama (zero camera-position baseline) correctly cannot be
  3D-reconstructed, no matter how good the 2D matching is; and a real
  multi-position reconstruction via `pycolmap.synthesize_dataset`
  (genuine parallax, genuine ground truth) proving `map_and_diagnose` and
  `evaluate_registration_quality` work correctly end-to-end.

**M5 -- Training (`vine360/training/`), adapter architecture only, per
an explicit choice not to install a backend or spend GPU time yet.**
- `config.py` (typed run config + `runs/<run-id>/` layout wiring),
  `adapter.py` (the backend contract), `nerfstudio_adapter.py` (a
  concrete implementation). **Unlike every other adapter in this
  project, `nerfstudio_adapter.py`'s CLI flags are unverified** -- no
  Nerfstudio install exists to confirm them against, unlike ffmpeg/
  pycolmap/SAM 3, which were each checked against real installed source
  or a real live call. See ADR 0009.

**M6 -- Desktop UI (`vine360/gui/`), started.** Built two full PySide6
previews to resolve the layout question directly rather than guessing:
`wizard_preview.py` (linear QWizard) and `dashboard_preview.py`
(persistent sidebar + stage panels). The user compared both running
side by side and chose the dashboard -- "I prefer the dashboard. I like
the more visual approach" (saved to memory). `main_window.py` is the real
app going forward: its Project and Import panels are genuinely wired to
`vine360.project`/`vine360.ingest.sources` (create/open a project on
disk, register a real source with ffprobe metadata + checksum + a
capture-group tag), verified end-to-end with an offscreen smoke test
(mocked file dialogs, a real synthetic clip, a real add_source call).
Frames/Projection/Masks/Pose panels show real preset
enums/computed values (e.g. the actual cubemap face count) but their
action buttons are disabled with an explanatory tooltip -- wiring those
to actually execute is intentionally deferred (see "Exact next task").
(Training's two placeholder panels were later removed from the sidebar
entirely rather than wired up -- see docs/adr/0025.)
The GUI calls the same typed library functions the CLI calls (not
CLI-as-subprocess) -- this still satisfies the doc's "every UI action
maps to a reproducible command" intent, since the action is backed by one
well-defined function either surface can call; it just means the CLI
isn't literally shelled out to.

**Export (`vine360/export/postshot.py`), new.** Bundles a completed SfM
run's poses/images/masks into a folder for import into Postshot (or any
COLMAP-based external trainer) -- the GUI's Export panel is now real
(pick a run, choose an output folder, export in the background). Found
and fixed a real pre-existing bug while building it: `sfm_runs.
selected_model` was recording the run's own `run_id`, not the COLMAP
model directory actually written -- `pycolmap.incremental_mapping`
writes every candidate reconstruction it finds under `sfm/sparse/<key>/`,
not just the best one, so this field is the only way to know which
subdirectory is real. See ADR 0018. Follow-up (ADR 0019) found a second,
worse bug behind it: every run shared the same `sfm/sparse/` directory,
so a later run could silently overwrite an earlier one's files --
confirmed for real on `/mnt/e/TEST` (3 of 4 historical runs
unrecoverable). Fixed going forward (each run gets `sfm/sparse/<run_id>/`
now) and added `vine360/sfm/repair_selected_model.py` (+ an Export-panel
button) to recover what's still recoverable from before the fix.

**Queue fixes (ADR 0030).** Three real bugs found using the persisted
processing queue (ADR 0023) on a live project: Cancel All only relabeled
pending jobs `SKIPPED` in place instead of actually clearing them, and
left a running job untouched with no user-visible explanation; and four
stage panels (Projection, Masks, Pose estimation, Export) populated their
source dropdowns with an INNER JOIN against the upstream table, silently
hiding any source that didn't have the prerequisite yet -- so the queue's
own prerequisite auto-insertion (queuing Masks on a source with no
projected views yet, auto-queuing Projection ahead of it) was unreachable
from the GUI. Both fixed; see ADR 0030 for the full design.

**RealityScan camera-priors export (ADR 0033).** While diagnosing a real
RealityScan export's ground-level vineyard-row footage completely
failing to register (0/1200 images, vs. 1118/1200 for the same project's
aerial pass), confirmed against each tool's real published docs that
RealityScan has a genuine "Camera Priors" pre-alignment feature (no
Postshot equivalent exists). `export_for_realityscan` can now optionally
also write a position-only `CameraPriors.csv` alongside the images,
drawn from vine360's own already-completed SfM run when one exists (GUI:
an "Include camera priors CSV" checkbox on the Export panel, RealityScan
mode only) -- for import via RealityScan's own WORKFLOW -> Import
Metadata -> Trajectory. No pose-composition math needed even for the
native-equirectangular SfM engine: verified for real that every cube
face shares its frame's own optical center, so a face's position is
identical to its frame's registered position.

**Frame sets, Data manager, and GUI fixes (ADRs 0034, 0035).** A source can now have
several extraction configs side by side (e.g. every 0.5s and every 1s), each a *frame set*
carried independently through projection, masking, SfM and export. Every stage dropdown lists
frame sets. Ids are deterministic per config, so re-extracting the same config replaces just
that set, and the queue can target a set before it's extracted. Existing projects migrate on
open: each source's frames become a frame set whose id is the source_id, with no file moved
(checked on a copy of a real project's database). SfM now hands COLMAP an explicit image list
and keeps a per-run database. A new **Data manager** tab lists sources (read-only) and each
frame set's frames/projections/masks, SfM runs and export folders, with sizes on disk, and
deletes them with full cascade. Mask files now cascade too; they used to be left behind.
Also:
- The Frames interval box shows the selected preset's interval.
- Export resets a manually pinned output folder when another project is opened.
- Export's folder dialog starts in the nearest existing folder, not the working directory.

**SphereSfM engine and pinhole export of 360 runs (ADR 0036).** SphereSfM is built from
source (CPU-only, `~/.local/spheresfm`) and is now a runnable Pose-estimation engine on
equirectangular frame sets, from the Pose tab or the queue. Its adapter is verified against the
real binary. Both 360 engines (SphereSfM and pycolmap EQUIRECTANGULAR) now export to Postshot
as ordinary pinhole cameras over vine360's own projected views, with masks, instead of raw
equirect frames. RealityScan camera priors work for both engines too. The frame-to-view pose
conversion is pinned by real reconstructions of a synthetic ray-cast 360 room: median 0.11 px
reprojection agreement, plus a cross-check against SphereSfM's own cube-face exporter.

**Verbose progress and Activity log (ADR 0037).**
- Every SfM step now reports counted, labelled progress. SphereSfM's live log is streamed and
  parsed, and each step's log is kept in the run folder. pycolmap's native log lines are
  captured during each call, and its mapper registration callbacks are used.
- Every panel's progress area has a timestamped Details log and a per-step ETA.
- A new Activity log tab lists every operation run in the project, in order, with its exact
  parameters, timing, outcome, and whether it was started by hand or by the queue. It can be
  exported to CSV.
- Verbose progress and activity logging are now standing requirements for new features
  (CLAUDE.md). pycolmap and SphereSfM both use the GPU (`sfm-cuda` extra; CUDA build of
  SphereSfM).

**Layered masks, overexposure layer, review fixes (ADR 0038).**
- Each mask class (sky, person, overexposure, excluded view) is its own layer under
  `masks/classes/<view_id>/`, merged into `masks/keep` on every change and re-checked before
  SfM and export.
- A layer can be regenerated, switched on/off per view or per set, or hand-edited outside the
  app without rebuilding the others.
- The new overexposure layer masks large fully clipped regions and their bloom while keeping white
  surfaces. It was checked by eye on real FMC-Tunnel1 views.
- Classical sky is now optional.
- Review keeps the selected face across frames and shows a colour overlay.
- "Flag for review" is replaced by "Exclude this view" and a list of suspicious views (coverage
  jumps compared with the same face in nearby frames).

**SfM advanced options (ADR 0040).**
- The Pose panel has a collapsible "Advanced options" section with 45 options for both engines.
  Groups: camera, features, matching strategy (sequential / exhaustive / vocabulary tree, loop
  detection), mapping (incremental or GLOMAP global on pycolmap), and quality-check thresholds.
- The form is generated from `vine360.sfm.options.OPTION_SPECS`, hides what the chosen engine
  can't use, and has presets and inline validation.
- Default options are verified to equal pycolmap's own defaults and SphereSfM's `-h` defaults,
  so a default run behaves exactly as before.
- Options are stored with each run and in queued jobs, and the panel starts from the latest run's
  options.

**View directions and the direction picker (ADR 0041).**
- Projection can render any of 26 named directions: yaw every 45° on a horizon ring, a
  tilted-down ring and a tilted-up ring (tilt adjustable, 45° by default), plus the two poles.
  The six original faces keep their exact rotations.
- The Projection panel's checkboxes are replaced by a picker drawn on the current frame's
  panorama:
  - clickable markers, with each selected view's footprint outlined;
  - a live hover preview of the view;
  - presets, including a vine-row tunnel preset (diagonal tilted-down views that see a row wall
    and the floor together);
  - a views × frames image count.
- Not yet tried on a real tunnel capture through SfM -- whether the tunnel preset actually
  registers better than the cardinal faces is still to be measured.

## Environment notes (this dev machine)

- No system `pip`/`venv` (Debian's `python3-venv`/`python3-pip` aren't
  installed, and there's no passwordless sudo to add them) -- worked
  around with `venv --without-pip` + a manually bootstrapped `get-pip.py`,
  in a venv living outside the repo (`~/.venvs/vine360`) because `/mnt/e`
  is a Windows-mounted drive that can't hold the symlinks `venv` needs.
  See ADR 0005.
- Real internet access exists, which is what made the rest of this
  possible: `numpy`/`Pillow`/`scipy`/`pycolmap`/`torch`/`transformers`/
  `static-ffmpeg` all installed and were exercised for real. See ADR 0006.
- This machine has a real NVIDIA GPU passed through into WSL2 (confirmed
  via `nvidia-smi` and `torch.cuda.is_available()`), matching the
  handover doc's own "Secondary environment: WSL2" line -- this looks
  like the actual target dev machine, not a disposable sandbox.
- `facebook/sam3` requires **manual** Meta approval (confirmed live via
  `HfApi().model_info(...).gated == "manual"`), not just a click-through
  license. **Approved and authenticated as of 2026-09-19** (`hf auth
  login` -- the modern replacement for the now-deprecated
  `huggingface-cli`); real SAM 3 inference now runs on this machine's GPU
  and is covered by a real, passing test
  (`test_segment_finds_sky_on_a_real_image`). Surfaced and fixed one real
  gap: `Sam3ImageProcessor` needs `torchvision`, not installed
  automatically alongside `torch`/`transformers`.
- Real capture equipment: an Insta360 (ground) and an **Antigravity A1**,
  which is itself an 8K 360-degree drone, not a conventional perspective
  drone as the handover doc's "mixed capture" milestone assumed -- see
  ADR 0010. Real sample footage exists at
  `/mnt/e/11-9-26_EstoWines_Capture1/Equi/` (multi-GB clips, a `.gpx`
  flight-telemetry sidecar); nothing has been run against it yet, per an
  explicit "no runs yet" instruction.

## Decisions (see docs/adr/ for full reasoning)

0001: stdlib-first CLI/config (argparse/dataclasses over Typer/Pydantic).
0002: Runner + adapter contract for external tools.
0003: project layout, deferred index tables, sources-by-reference.
0004: M1 scope assumptions (input format, EXIF/GPS, compute environment).
0005: venv lives outside the repo, under `$HOME` (DrvFs symlink limits).
0006: real numpy/Pillow/pycolmap/ffmpeg once pip access existed after all.
0007: pycolmap (not a subprocess `colmap` binary) for the SfM adapter.
0008: SAM 3 as the real masking backend; classical sky as an interim
      fallback; reordered mask-build pipeline (dedupe before dilate).
      Updated 2026-09-19: real access confirmed, real inference tested.
0009: training stays adapter-only; Nerfstudio adapter is unverified.
0010: real capture kit is two 360-degree platforms (Insta360 ground +
      Antigravity A1 aerial), not 360 + conventional; capture_group is
      now a real, wired parameter on add_source.
0011: fixed re-extraction's UNIQUE constraint bug and a QThread
      use-after-free crash in the GUI's background-worker helper; GUI now
      runs under QT_QPA_PLATFORM=xcb (not wayland) for more reliable
      window stacking/dropdowns, using a user-space-extracted
      libxcb-cursor0 (no root needed).
0012: wired Projection, Masking and SfM into the GUI (new
      vine360.projection.generate / vine360.masking.build /
      vine360.sfm.project_run persistence-layer modules, three new index
      tables); unified every progress_callback to (message, current,
      total) with real ETAs; sidebar stage status is now derived live
      from the database (compute_stage_statuses) instead of tracked ad
      hoc; Project panel split into clearly separated create/open
      sections. Also fixed a real latent bug this surfaced:
      pycolmap.COLMAP_version/COLMAP_build are str attributes, not
      callables -- validate_installation() had been calling them as
      functions since ADR 0007, uncaught until now.
0034: frame sets -- several extraction configs per source, deterministic
      ids, legacy id == source_id migration, per-run COLMAP database.
0035: Data manager tab -- list everything derived, delete with cascade.
0037: verbose, counted progress for long operations (streamed tool logs,
      native log capture -- never DB polling) and a per-project activity log.
0036: SphereSfM engine (CLI via Runner, TXT models -- never pycolmap) and
      pinhole export of 360 runs via measured frame->view pose conversion.
0038: layered masks (per-class PNG + mask_layers row, composite keep-mask
      rebuilt on change), clipping-based overexposure layer, review that acts.
0040: SfM advanced options -- one stdlib SfmConfig for both engines, GUI form
      generated from OPTION_SPECS, defaults == each engine's own defaults.
0041: 26-direction view catalog (45° yaw steps, tilted rings, legacy six unchanged) and a
      graphical direction picker drawn over the frame's own panorama.

## Blockers / known gaps

- **Nerfstudio adapter is unverified** (see ADR 0009) -- run its `--help`
  output against a real install before trusting its flags.
- Near-duplicate frame filtering (M1, called for in section 4) still
  isn't implemented.
- Photo EXIF/GPS parsing for drone stills (ADR 0004) still isn't
  implemented -- and a GPX-sidecar reader (see ADR 0010) is now a better
  first georeferencing step than EXIF parsing, given the real sample
  data has a `.gpx` file, not JPEG GPS tags.
- The GUI (`vine360/gui/`) has no automated test *suite* yet -- every
  verification so far (including the full Project-through-SfM pipeline)
  was a manual offscreen smoke test run during development, not added to
  pytest. `pytest-qt` or similar isn't a dependency yet.
- No pipeline stage has been run against real footage. Real EstoWines
  sample data exists (see above); every test and smoke-test run so far
  uses synthetic clips/images, per the standing preference not to run
  multi-GB real files without it being explicitly asked for.
- Training has no GUI presence at all now -- its two placeholder panels
  (preset selection UI, monitor UI) were removed from the sidebar (ADR
  0025) rather than left disabled. The library code stays adapter-only,
  per the standing decision in ADR 0009; wiring a real backend remains
  future work, not a gap in what shipped here.

## Exact next task

With Project through Pose estimation now real and wired, reasonable next
directions (none started):

1. Add a `pytest-qt`-based automated GUI test suite -- codify the manual
   offscreen smoke tests (create -> import -> frames -> projection ->
   masks -> SfM, plus the remove-source/custom-interval/re-extraction
   paths) as real pytest tests instead of ad hoc scripts.
2. Training is out of the app's scope for now (ADR 0025 removed its GUI
   panels entirely). Wiring it for real would mean both revisiting ADR
   0009's "adapter only" decision and re-adding a GUI stage informed by
   whatever the real backend's actual UI needs turn out to be.
3. Georeferencing (M8): a GPX-sidecar reader (see ADR 0010) is a better
   first step than the still-unimplemented EXIF/GPS parsing, given real
   sample data already has `.gpx` files.
4. Near-duplicate frame filtering (M1 gap, still open).
5. Validate the real, wired pipeline against actual Insta360/Antigravity
   A1 footage -- explicitly on request given multi-GB file sizes.
