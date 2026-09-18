"""3DGS training backend adapter (handover doc, section 4, step 7 "Train").

Adapter-only: no training backend is installed or executed by this
codebase -- see docs/adr/0009. config.py holds the typed run configuration
and project run-directory layout; adapter.py is the backend contract;
nerfstudio_adapter.py is a concrete (unverified -- see its docstring)
implementation.
"""
