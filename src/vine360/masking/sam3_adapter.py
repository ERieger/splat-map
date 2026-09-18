"""SAM 3 masking adapter (handover doc, section 4, step 4 "Mask"; section 3's
Adapter rule: "validate installation, report version and capabilities...
classify failures").

SAM 3 is an in-process PyTorch model, not a subprocess CLI, so this follows
the adapter rule's spirit without going through vine360.runners.Runner
(which is for subprocess engines like ffmpeg/COLMAP). Method names and
signatures below (`Sam3Processor(images=..., text=...)`,
`post_process_instance_segmentation(outputs, target_sizes=...)` returning
`{"masks": ..., "boxes": ..., "scores": ...}`) were confirmed against the
actual installed `transformers` source
(transformers/models/sam3/processing_sam3.py), not guessed from
documentation, since the gated weights themselves haven't been
downloaded/run yet -- see docs/adr/0008.

facebook/sam3 requires manual approval on Hugging Face. Construction never
touches the network; only the first `segment()` call loads the model, and
a gated/unauthenticated repo is reported as `Sam3AccessDeniedError` rather
than an unhandled exception.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

MASK_TRUE = 255
MASK_FALSE = 0

SAM3_MODEL_ID = "facebook/sam3"


class Sam3AdapterError(Exception):
    pass


class Sam3NotInstalledError(Sam3AdapterError):
    pass


class Sam3AccessDeniedError(Sam3AdapterError):
    """The repo is gated and this environment isn't authenticated/approved."""


@dataclass
class Sam3Status:
    torch_installed: bool
    transformers_installed: bool
    cuda_available: bool
    torch_version: str | None
    transformers_version: str | None


def validate_installation() -> Sam3Status:
    torch_installed = importlib.util.find_spec("torch") is not None
    transformers_installed = importlib.util.find_spec("transformers") is not None
    torch_version = None
    cuda_available = False
    if torch_installed:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    transformers_version = None
    if transformers_installed:
        import transformers

        transformers_version = transformers.__version__
    return Sam3Status(
        torch_installed=torch_installed,
        transformers_installed=transformers_installed,
        cuda_available=cuda_available,
        torch_version=torch_version,
        transformers_version=transformers_version,
    )


class Sam3Adapter:
    """Text-prompted segmentation via facebook/sam3, loaded lazily."""

    def __init__(self, model_id: str = SAM3_MODEL_ID, device: str | None = None):
        status = validate_installation()
        if not (status.torch_installed and status.transformers_installed):
            raise Sam3NotInstalledError(
                "torch and transformers are required; install the 'masking' extra"
            )
        self.model_id = model_id
        self.device = device or ("cuda" if status.cuda_available else "cpu")
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub.errors import GatedRepoError
        from transformers import Sam3Model, Sam3Processor

        try:
            self._processor = Sam3Processor.from_pretrained(self.model_id)
            model = Sam3Model.from_pretrained(self.model_id)
        except (GatedRepoError, OSError) as exc:
            # transformers' from_pretrained doesn't reliably surface
            # huggingface_hub's GatedRepoError as its own type -- in
            # practice (confirmed against a real unauthenticated call) it
            # re-raises the underlying 401 as a plain OSError whose message
            # mentions the gate. Detect that case by message rather than
            # type so a real gating failure is classified correctly instead
            # of masquerading as a generic adapter error.
            if isinstance(exc, GatedRepoError) or "gated repo" in str(exc).lower():
                raise Sam3AccessDeniedError(
                    f"{self.model_id} is gated. Request access at "
                    f"https://huggingface.co/{self.model_id}, wait for manual approval, "
                    "then authenticate this environment with `huggingface-cli login` "
                    "before retrying."
                ) from exc
            raise
        self._model = model.to(self.device)
        self._model.eval()

    def segment(self, image: np.ndarray, prompts: dict[str, str]) -> dict[str, np.ndarray]:
        """image: (H, W, 3) uint8 RGB. prompts: {class_name: text_prompt},
        e.g. {"person": "person", "sky": "sky"}. Returns
        {class_name: (H, W) uint8 mask, 255=matched}."""
        self._load()
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        height, width = image.shape[0], image.shape[1]
        masks: dict[str, np.ndarray] = {}
        for class_name, prompt in prompts.items():
            inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            result = self._processor.post_process_instance_segmentation(
                outputs, target_sizes=[(height, width)]
            )[0]
            instance_masks = result["masks"]
            combined = np.zeros((height, width), dtype=bool)
            for instance_mask in instance_masks:
                combined |= instance_mask.detach().cpu().numpy().astype(bool)
            masks[class_name] = np.where(combined, MASK_TRUE, MASK_FALSE).astype(np.uint8)
        return masks
