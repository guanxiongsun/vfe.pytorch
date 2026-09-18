"""vfe — pure-PyTorch video object detection (MAMBA, STPN, SELSA).

This package is the rewrite target: it replaces the vendored mmdetection 2.19.1
tree at ``mmdet/`` with plain ``torch`` + ``torchvision.ops`` code. During the
port, ``mmdet/`` stays in the repo as a reference oracle (see REWRITE_PLAN.md).
"""

__version__ = "0.1.0"
