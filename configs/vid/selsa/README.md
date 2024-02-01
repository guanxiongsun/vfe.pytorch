# Sequence Level Semantics Aggregation for Video Object Detection

## Reference

- [mmtracking/configs/vid/selsa](https://github.com/open-mmlab/mmtracking/blob/master/configs/vid/selsa/README.md)

## Results and models on ImageNet VID dataset

We observe around 1 mAP fluctuations in performance, and provide the best model.

|      Method       | Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP@50 |                           Config                           |                                                                                                                                                                         Download                                                                                                                                                                         |
| :---------------: | :-------: | :-----: | :-----: | :------: | :------------: | :-------: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       SELSA       | R-50-DC5  | pytorch |   7e    |   3.49   |      7.5       |   78.4    |   [config](selsa_faster_rcnn_r50_dc5_1x_imagenetvid.py)    |   [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835.log.json)   |
|       SELSA       | R-101-DC5 | pytorch |   7e    |   5.18   |      7.2       |   81.5    |   [config](selsa_faster_rcnn_r101_dc5_1x_imagenetvid.py)   | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724.log.json) |
|       SELSA       | X-101-DC5 | pytorch |   7e    |   9.15   |       -        |   83.1    |   [config](selsa_faster_rcnn_x101_dc5_1x_imagenetvid.py)   | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_x101_dc5_1x_imagenetvid/selsa_faster_rcnn_x101_dc5_1x_imagenetvid_20210825_205641-10252965.pth) \| [log](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_x101_dc5_1x_imagenetvid/selsa_faster_rcnn_x101_dc5_1x_imagenetvid_20210825_205641.log.json) |
| SELSA <br> (FP16) | R-50-DC5  | pytorch |   7e    |   2.71   |       -        |   78.7    | [config](selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid.py) |                                            [model](https://download.openmmlab.com/mmtracking/fp16/selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid_20210728_193846-dce6eb09.pth) \| [log](https://download.openmmlab.com/mmtracking/fp16/selsa_faster_rcnn_r50_dc5_fp16_1x_imagenetvid_20210728_193846.log.json)                                            |

Note:

- `FP16` means Mixed Precision (FP16) is adopted in training.