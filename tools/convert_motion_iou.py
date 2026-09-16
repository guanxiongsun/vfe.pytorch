"""Convert the ImageNet VID val motion-IoU table from MATLAB to a flat numpy file.

One-off, run in the legacy env (it needs scipy):
    conda run -n vfe --no-capture-output python tools/convert_motion_iou.py

Reads  mmdet/datasets/mamba/vid_groundtruth_motion_iou.mat  (FGFA's table: one
       entry per ground-truth object per val frame, in val-frame order)
Writes vfe/evaluation/vid_motion_iou.npz  with
       values   float64, every frame's entries concatenated
       offsets  int64, len(frames) + 1; frame i is values[offsets[i]:offsets[i+1]]

The per-frame lists are built exactly as mmdet's vid_eval.py builds them,
*including* its placeholder: a frame with no objects has one empty MATLAB entry,
which becomes a motion IoU of 0. Those zeros are not dropped, because the
evaluator counts them when weighting false positives (``empty_weight``).
The flat layout avoids a ragged object array, which numpy >= 1.24 refuses to
build implicitly, and loads without pickle or scipy.
"""

import os

import numpy as np
import scipy.io as sio

SRC = "mmdet/datasets/mamba/vid_groundtruth_motion_iou.mat"
DST = "vfe/evaluation/vid_motion_iou.npz"


def legacy_per_frame_lists(mat_path):
    """Verbatim from mmdet/datasets/mamba/vid_eval.py, minus the np.array wrap."""
    motion_ious = sio.loadmat(mat_path)
    return [
        [
            motion_ious["motion_iou"][i][0][j][0]
            if len(motion_ious["motion_iou"][i][0][j]) != 0
            else 0
            for j in range(len(motion_ious["motion_iou"][i][0]))
        ]
        for i in range(len(motion_ious["motion_iou"]))
    ]


def main():
    frames = legacy_per_frame_lists(SRC)
    lengths = np.array([len(f) for f in frames], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    values = np.array([float(v) for f in frames for v in f], dtype=np.float64)
    assert len(values) == offsets[-1]

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    np.savez_compressed(DST, values=values, offsets=offsets)
    print(f"{len(frames)} frames, {len(values)} entries -> {DST} "
          f"({os.path.getsize(DST) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
