# TDViT

**[TDViT: Temporal Dilated Video Transformer for Dense Video Tasks](https://arxiv.org/abs/2402.09257)**
(ECCV 2022), implemented from its paper and the backbone of the authors' CVPR
2022 supplementary code; its repository was never released.

## Results

ImageNet VID validation, Faster R-CNN with an FPN, 3 epochs at batch 8 from
ImageNet-1K Swin-T weights with Swin's augmentation; COCO-style AP as the
paper's Table 2 reports it, and the VID metric:

| Backbone | | AP | AP50 | AP75 | VID AP50 | paper (AP / AP50 / AP75) |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: |
| Swin-T | | 50.2 | 79.0 | 55.4 | 79.5 | 47.1 / 77.2 / 51.5 |
| TDViT-T | as published | 46.0 | 75.7 | 49.4 | 76.2 | 49.1 / 78.5 / 52.7 |
| | joint attention | 50.6 | 80.3 | 56.0 | 80.8 | |
| TDViT-T+ | joint attention, new blocks copied | **51.4** | **80.8** | **57.3** | **81.4** | 50.9 / 79.9 / 55.7 |

A second seed gives Swin-T 50.6 / 79.0 and TDViT-T 50.6 / 79.7: TDViT-T's
AP50 gain holds (+0.7 to +1.3), its AP gain (0.0 to +0.4) is within the
run-to-run spread of 0.4. Trained instead with v1's plain pipeline (resize
and flip, the `*_v1aug.py` configs), both tiny models score higher and the gap
widens: Swin-T 51.1 / 79.4 / 57.0, TDViT-T 51.8 / 81.2 / 57.6 (VID AP50 79.9
and 81.7); TDViT-T+ scores 51.7 / 81.0 / 57.5 there, no better than TDViT-T.
No checkpoint is published yet.

As published, a temporal block attends from the frame to a reference frame
alone, which gives up the frame's own spatial attention in half of TDViT-T's
blocks: trained here next to a Swin-T on the same recipe, it trains worse and
tests 4.2 AP lower, most of it on fast objects. With one change — each
temporal block attends over its own window and the reference's together
(`attention='joint'`, no new parameter) — TDViT-T passes both the Swin-T and
the paper's TDViT-T, at Swin-T's size and, with `fused_attention=True`, its
speed (52.0 against 51.8 FPS on a GH200). TDViT-T+'s two extra blocks have no
ImageNet weights and learn nothing at this learning rate from a random start;
copied from the pretrained blocks before them (`extra_init='copy'`) they add
0.8 AP on Swin's augmentation and nothing on the plain recipe — a smaller and
less certain gain than the paper's 1.8. The Swin-T trained here is 3.1 AP
stronger than the paper's. SELSA on TDViT-T reproduces the paper's Table 3
(VID AP50 83.8 against 83.9), though SELSA on Swin-T, which the paper does not
report, scores 84.5. The small and base variants are configured but not yet
tuned. How every number was reached:
[docs/tdvit-plan.md](../../../docs/tdvit-plan.md).

## Configs

Faster R-CNN with an FPN on ImageNet VID, 3 epochs at batch 8, AdamW at 2.5e-5
from ImageNet-1K Swin weights, with Swin's augmentation unless the name says
otherwise:

| Config | Model |
| :-- | :-- |
| [`frcnn_swint_fpn_3x.py`](frcnn_swint_fpn_3x.py) | Swin-T, the single-frame baseline (Table 2) |
| [`tdvit_t_frcnn_fpn_3x.py`](tdvit_t_frcnn_fpn_3x.py) | TDViT-T as published: a temporal block attends to its reference alone |
| [`tdvit_t_joint_frcnn_fpn_3x.py`](tdvit_t_joint_frcnn_fpn_3x.py) | **TDViT-T with joint attention** — the one to use |
| [`tdvit_tplus_frcnn_fpn_3x.py`](tdvit_tplus_frcnn_fpn_3x.py) | TDViT-T+ as published: two more temporal blocks in stage 3, from torch's initialisation |
| [`tdvit_tplus_joint_frcnn_fpn_3x.py`](tdvit_tplus_joint_frcnn_fpn_3x.py) | TDViT-T+ with joint attention, the two new blocks copied from the pretrained blocks before them |
| [`frcnn_swins_fpn_3x.py`](frcnn_swins_fpn_3x.py), [`tdvit_s_joint_frcnn_fpn_3x.py`](tdvit_s_joint_frcnn_fpn_3x.py) | Swin-S, and TDViT-S with joint attention |
| [`frcnn_swinb_fpn_3x.py`](frcnn_swinb_fpn_3x.py), [`tdvit_b_joint_frcnn_fpn_3x.py`](tdvit_b_joint_frcnn_fpn_3x.py) | Swin-B and TDViT-B: they build, but were never trained |
| [`frcnn_swint_fpn_3x_v1aug.py`](frcnn_swint_fpn_3x_v1aug.py), [`tdvit_t_joint_frcnn_fpn_3x_v1aug.py`](tdvit_t_joint_frcnn_fpn_3x_v1aug.py), [`tdvit_tplus_joint_frcnn_fpn_3x_v1aug.py`](tdvit_tplus_joint_frcnn_fpn_3x_v1aug.py) | the three tiny models on v1's plain pipeline (resize to 600, flip), which scores higher |
| [`selsa_swint_fpn_3x.py`](selsa_swint_fpn_3x.py), [`selsa_tdvit_t_joint_fpn_3x.py`](selsa_tdvit_t_joint_fpn_3x.py) | SELSA\* (Table 3: SELSA with RDN's top-75 reference proposals) on Swin-T and on TDViT-T with joint attention |
| [`selsa_tdvit_t_fpn_3x.py`](selsa_tdvit_t_fpn_3x.py) | SELSA\* on TDViT-T as published; never trained |

## Test-time settings

TDViT's test set keeps each video's frames in order, first frame first: the
detector resets the backbone's memories at `frame_id == 0` and fills them as
the video goes on. The memory is a test-time setting of one trained model, so
one checkpoint gives the paper's Table 8 and more through `--cfg-options`:
`model.detector.backbone.memory_sampling=nms` (or `patch_shuffle`,
`channel_shuffle`), `model.detector.backbone.memory_reuse=1` (a new reference
every frame), `model.detector.backbone.temporal_dilations=(1,2,4,8)`, and
`model.online=False` (the memory off: the weights as a still-image detector).
`model.detector.backbone.fused_attention=True` runs the attention through
PyTorch's fused kernel, the same detections faster; it is off by default so
Swin stays bit-identical to its reference.
