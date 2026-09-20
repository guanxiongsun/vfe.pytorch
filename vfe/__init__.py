"""vfe — pure-PyTorch video object detection.

MAMBA and STPN on ImageNet VID, implemented on plain ``torch`` and
``torchvision.ops``: no mmcv, mmdet or mmengine at runtime. The original
mmdetection 2.19.1 implementation this was ported from is preserved at the
``v1.0.0`` tag and is used as the reference the port is checked against
(see docs/parity.md).
"""

__version__ = "2.0.0"
