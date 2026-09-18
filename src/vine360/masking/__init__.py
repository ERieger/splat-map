"""People/sky segmentation and mask semantics (handover doc, section 4,
step 4 "Mask").

semantics.py: mask value conventions, morphology, keep/exclude/COLMAP mask
export -- no model dependency. classical_sky.py: a weights-free sky
heuristic usable while SAM 3 access is pending. sam3_adapter.py: the real
SAM 3 backend (gated; see docs/adr/0008).
"""
