# The Faster R-CNN baseline of the paper's Table 3 (75.4): MAMBA with neither
# level, trained on the same key frames with the same recipe. The reference
# frames are loaded and ignored.
_base_ = ['./mamba_pix_r101_dc5_3x.py']

model = dict(pixel=None)
