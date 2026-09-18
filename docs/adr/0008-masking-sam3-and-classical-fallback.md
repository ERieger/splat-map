# 0008. Masking: SAM 3 as the real backend, classical sky heuristic as a
fallback while access is pending, and a reordered mask-build pipeline

## Status

Accepted.

## Context

Section 4, step 4 ("Mask") requires segmenting person and sky, dilating
the exclusion mask, removing tiny components, and producing both class
masks and a final keep-mask. `facebook/sam3` (the doc's named model,
section 13) is gated on Hugging Face with **manual** approval, confirmed
live via `HfApi().model_info("facebook/sam3").gated == "manual"` -- not a
click-through license, an actual review queue. The user chose "download
SAM 3 now" over building adapter-only, but manual approval can't be forced
or predicted, so the adapter had to be built and tested without weights
in hand yet.

## Decision

- **`vine360/masking/sam3_adapter.py`** implements the real backend against
  `transformers`' `Sam3Model`/`Sam3Processor`. The exact API calls
  (`processor(images=..., text=...)`,
  `post_process_instance_segmentation(outputs, target_sizes=...)` ->
  `{"masks": (N,H,W), "boxes": ..., "scores": ...}`) were confirmed by
  reading the actual installed `transformers` source
  (`transformers/models/sam3/processing_sam3.py`), not guessed from
  documentation -- the doc/blog description of the model doesn't specify
  method signatures precisely enough to trust blind.
- Construction (`Sam3Adapter()`) never touches the network; only
  `segment()` loads the model. A real, live call against the gated,
  unauthenticated repo was made during development and initially raised a
  bare `OSError` instead of `huggingface_hub`'s `GatedRepoError` --
  `transformers`' `from_pretrained` doesn't reliably preserve the
  original exception type. The adapter now classifies by message content
  ("gated repo" in the text) as well as type, confirmed against the real
  endpoint to raise `Sam3AccessDeniedError` with actionable instructions.
- **`vine360/masking/classical_sky.py`** is a coarse, weights-free
  brightness/blue-dominance/top-of-frame heuristic, explicitly framed as a
  stand-in while SAM 3 access is pending (or a cross-check afterward, per
  section 10's "Sky quality" risk). There is no equivalent fallback for
  person masks -- no color/brightness heuristic reliably separates a
  person from vineyard foliage, trellis or equipment, so person exclusion
  genuinely requires SAM 3 or an equivalent model.
- **`vine360/masking/semantics.py`** builds the exclusion mask by removing
  tiny components *before* dilating, the reverse of the doc's literal
  step order ("Dilate exclusion masks slightly, remove tiny components").
  Dilating first can fuse small noise specks into real regions, after
  which a component-size filter can no longer isolate and remove them.

## Consequences (original, at write time)

- Real SAM 3 inference is untested (no weights available yet); once
  access is granted and `huggingface-cli login` is run, replace
  `tests/test_masking_sam3_adapter.py`'s
  `test_segment_raises_access_denied_when_ungated_access_missing` with a
  real segmentation assertion against a known image, and re-verify the
  `post_process_instance_segmentation` output shape assumptions above
  against actual model output rather than the source code alone.
- If SAM 3's released checkpoint turns out to live under a different
  model id (the transformers docstring example uses
  `facebook/sam3-base`, not `facebook/sam3`), update `SAM3_MODEL_ID`
  accordingly -- `Sam3Adapter(model_id=...)` already supports overriding
  it without a code change.

## Update, 2026-09-19: access granted, real inference confirmed

Meta approved access the same day; `hf auth login` was run (token stored
outside the repo, never in chat/logs), and `Sam3Model`/`Sam3Processor`
loaded and ran for real on the GPU. `model.safetensors` (3.44 GB) needed
roughly 15-20 minutes to download once; it's now cached under
`~/.cache/huggingface/hub/models--facebook--sam3`.
`test_segment_finds_sky_on_a_real_image` replaced the access-denied test
and passes: on a synthetic sky/ground image, real SAM 3 correctly
concentrates the sky mask in the top rows with plausible coverage. The
`post_process_instance_segmentation` output shape assumed from source
inspection was correct -- no adapter code changes were needed.

One dependency gap surfaced and was fixed: `Sam3ImageProcessor` requires
`torchvision`, which isn't pulled in by `torch`/`transformers` alone.
Install it alongside them (same CUDA-matched index URL as `torch`).

`SAM3_MODEL_ID = "facebook/sam3"` (not `facebook/sam3-base`) is confirmed
correct as used.
