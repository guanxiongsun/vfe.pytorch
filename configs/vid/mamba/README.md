# MAMBA

**[MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection](https://arxiv.org/abs/2401.09923)**
(AAAI 2021), ported from the released code.

## Results

ImageNet VID validation, AP50, measured on 4× GH200:

| | released checkpoint, evaluated here | trained here, from scratch | originally published |
| :-- | :--: | :--: | :--: |
| MAMBA, ResNet-101-DC5 | 83.80 | 84.06 | 83.82 |

The released checkpoint scores 65.3 / 83.8 / 89.5 AP50 on fast / medium /
slow objects; it loads into this code unchanged:
[`mamba_r101_dc5_6x`](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/mamba_r101_dc5_6x).
For reference, a single-frame Faster R-CNN on ResNet-101 scores 76.7
([MEGA's table](https://github.com/Scalsol/mega.pytorch#main-results)).

## Configs

| Config | Model |
| :-- | :-- |
| [`mamba_r101_dc5_6x.py`](mamba_r101_dc5_6x.py) | MAMBA, ResNet-101-DC5, 6 epochs: the released model (the instance level) |
| [`mamba_r101_dc5_3x.py`](mamba_r101_dc5_3x.py) | the same, 3 epochs |
| [`frcnn_r101_dc5_3x.py`](frcnn_r101_dc5_3x.py) | the single-frame Faster R-CNN baseline (the paper's Table 3: 75.4) |
| [`mamba_full_r101_dc5_3x.py`](mamba_full_r101_dc5_3x.py), [`mamba_full_r101_dc5_6x.py`](mamba_full_r101_dc5_6x.py) | the paper's full model: the pixel level added (Table 3: 84.6) |
| [`mamba_pix_r101_dc5_3x.py`](mamba_pix_r101_dc5_3x.py) | the pixel level alone (Table 3: 81.8) |
| [`mamba_pix_neck_r101_dc5_3x.py`](mamba_pix_neck_r101_dc5_3x.py) | the pixel level on the ChannelMapper's 512-channel output |

## Notes

**The published model did not train on the schedule its config describes.**
Its checkpoint records a 4-GPU run resumed on 8, and mmcv rescaled the
iteration count on resume, so epochs 1–3 ran at batch 4 and epochs 4–6 at batch
8. Read literally — batch 8 throughout — the config gives half the steps in
epochs 1–3 and scores 83.16. The 84.06 above follows the published model's own
schedule: train epochs 1–3 with half the batch, then resume.

```bash
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train \
    configs/vid/mamba/mamba_r101_dc5_6x.py --launcher pytorch \
    --accumulate 1 --max-epochs 3 --work-dir WORK_DIR         # batch 4
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train \
    configs/vid/mamba/mamba_r101_dc5_6x.py --launcher pytorch \
    --accumulate 2 --resume-from auto --work-dir WORK_DIR     # batch 8
```

**The pixel level.** The released model is the paper's instance-level variant
(Table 3, "Ours_ins": 83.7), which the numbers above reproduce. The paper's
full model also enhances the feature map before the RPN, and that pixel level
was never released ([#4](https://github.com/guanxiongsun/vfe.pytorch/issues/4)).
It is implemented here: after one epoch the full model scores 75.6 AP50
against 72.1 for the instance level alone
([docs/mamba-pixel-plan.md](../../../docs/mamba-pixel-plan.md)). There is no
full-schedule checkpoint of it yet.
