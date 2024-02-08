# Video Feature Enhancement with PyTorch

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This repo contains the code for the paper:
[MAMBA](https://arxiv.org/abs/2401.09923), [STPN](https://arxiv.org/abs/2402.02574), TDViT, EOVOD


Additionally, we provide archive files of two widely-used datasets, [ImageNetVID](https://huggingface.co/datasets/guanxiongsun/imagenetvid/tree/main) and GOT-10K. The official links of these datasets are not accessible or deleted. We hope these resources can help future research.

## Progress

- [x] [MAMBA](https://arxiv.org/abs/2401.09923)
- [x] [STPN](https://arxiv.org/abs/2402.02574)
- [ ] TDViT
- [ ] EOVOD

## Main Results

|       Model        |  Backbone  | AP50 | AP (fast) | AP (med) | AP (slow) |                                             Link                                             |
| :----------------: | :--------: | :--: | :-------: | :------: | :-------: | :------------------------------------------------------------------------------------------: |
|     FasterRCNN     | ResNet-101 | 76.7 |   52.3    |   74.1   |   84.9    | [model](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view?usp=sharing), [reference](https://github.com/Scalsol/mega.pytorch/tree/master?tab=readme-ov-file#main-results)|
|     SELSA          | ResNet-101 |  81.5  |    --     |    --    |    --     | [model](https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r101_dc5_1x_imagenetvid/selsa_faster_rcnn_r101_dc5_1x_imagenetvid_20201218_172724-aa961bcc.pth), [reference](https://github.com/open-mmlab/mmtracking/tree/master/configs/vid/selsa) |
|     MEGA     |   ResNet-101  |  82.9	|62.7	|81.6	|89.4  | [model](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view?usp=sharing), [reference](https://github.com/Scalsol/mega.pytorch/tree/master) |
|     **MAMBA**     | ResNet-101 |  83.8 | 65.3 | 83.8 | 89.5 | [config](configs/vid/mamba), [model](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/mamba_r101_dc5_6x), [paper](https://arxiv.org/abs/2401.09923)|
|     **STPN**      | Swin-T |  85.2 | 64.1 | 84.1 | 91.4 | [config](configs/vid/stpn), [model](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/stpn_swint_adam_9x), [paper](https://arxiv.org/abs/2402.02574)|


## Installation
The code are tested with the following environments:

### Tested environments:

- python 3.8
- pytorch 1.10.1
- cuda 11.3
- mmcv-full 1.3.17

### Option 1: Step-by-step installation

```bash
conda create --name vfe -y python=3.8
conda activate vfe

# install PyTorch with cuda support
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# install mmcv-full 1.3.17
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# install other requirements
pip install -r requirements.txt

# install mmpycocotools
pip install mmpycocotools
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

## Data preparation

### Download ImageNetVID (Video Object Detection) Dataset

The original links of ImageNetVID dataset are either broken or unavailible. Here, we provide the new link to download the file for the furture reference of the community. Please download ILSVRC2015 DET and ILSVRC2015 VID datasets from this [LINK](https://huggingface.co/datasets/guanxiongsun/imagenetvid/tree/main). 

After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

    ./data/ILSVRC/
    ./data/ILSVRC/Annotations/DET
    ./data/ILSVRC/Annotations/VID
    ./data/ILSVRC/Data/DET
    ./data/ILSVRC/Data/VID
    ./data/ILSVRC/ImageSets

**Note**: List txt files under `ImageSets` folder can be obtained from
[here](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

### Convert Annotations

We use [CocoVID](mmdet/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase. In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```bash
# ImageNet DET
python ./tools/convert_datasets/ilsvrc/imagenet2coco_det.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

# ImageNet VID
python ./tools/convert_datasets/ilsvrc/imagenet2coco_vid.py -i ./data/ILSVRC -o ./data/ILSVRC/annotations

```

## Usage

### Inference

This section will show how to test existing models on supported datasets.
The following testing environments are supported:

- single GPU
- single node multiple GPU

During testing, different tasks share the same API and we only support `samples_per_gpu = 1`.

You can use the following commands for testing:

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [--checkpoint ${CHECKPOINT_FILE}] [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

Optional arguments:

- `CHECKPOINT_FILE`: Filename of the checkpoint. You do not need to define it when applying some MOT methods but specify the checkpoints in the config.
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `bbox` is available for ImageNet VID, `track` is available for LaSOT, `bbox` and `track` are both suitable for MOT17.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file
- `--eval-options`: If specified, the key-value pair optional eval cfg will be kwargs for dataset.evaluate() function, it’s only for evaluation
- `--format-only`: If specified, the results will be formatted to the official format.

#### Examples of testing VID model

Assume that you have already downloaded the checkpoints to the directory `work_dirs/`.

1. Test MAMBA on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   python tools/test.py configs/vid/mamba/mamba_r101_dc5_6x.py \
       --checkpoint work_dirs/mamba_r101_dc5_6x/epoch_6_model.pth \
       --out results.pkl \
       --eval bbox
   ```

2. Test MAMBA with 8 GPUs on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   ./tools/dist_test.sh configs/vid/mamba/mamba_r101_dc5_6x.py 8 \
       --checkpoint work_dirs/mamba_r101_dc5_6x/epoch_6_model.pth \
       --out results.pkl \
       --eval bbox
   ```

### Training

#### Training on a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file or via CLI argument `--work-dir`.

#### Training on multiple GPUs

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

#### Examples of training VID model

1. Train MAMBA on ImageNet VID and ImageNet DET with single GPU, then evaluate the bbox mAP at the last epoch.

   ```shell
   python tools/train.py configs/vid/mamba/mamba_r101_dc5_6x.py 
   ```

2. Train MAMBA on ImageNet VID and ImageNet DET with 8 GPUs, then evaluate the bbox mAP at the last epoch.

   ```shell
   ./tools/dist_train.sh configs/vid/mamba/mamba_r101_dc5_6x.py 8
   ```

## Reference

The codebase is implemented based on two popular open-source repos:
 [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmtracking](https://github.com/open-mmlab/mmtracking) in [PyTorch](https://pytorch.org/).