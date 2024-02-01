# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (init_dist, wrap_fp16_model)

from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector, build_model
from mmdet.utils import get_root_logger
from mmdet.utils.logger import get_speed_tester
import timeit
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create logger
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        args.work_dir = cfg.work_dir

    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}_speed_tester.log')
    speed_tester = get_speed_tester(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    # for video models
    if 'detector' in cfg.model.keys():
        # multi-frame video model
        if cfg.model.get('type', False) in ("SELSA", "MAMBA"):
            model = build_model(cfg.model)

        # single-frame video base model
        else:
            cfg.model = cfg.model.detector
            model = build_detector(
                cfg.model,
                test_cfg=cfg.get('test_cfg'))

    # for single-frame models
    else:
        model = build_detector(
            cfg.model,
            test_cfg=cfg.get('test_cfg'))

    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0])

    # speed test
    model.eval()
    data = []
    total_test_iterations = 500
    for i, _data in enumerate(data_loader):
        if len(data) == 2:
            break
        data.append(_data)

    # overall speed test
    speed_tester.reset()
    overall_speed_test(model, data, total_test_iterations, speed_tester)

    # # part speed test
    # speed_tester.reset()
    # partial_speed_test(model, data, total_test_iterations, speed_tester)


def overall_speed_test(model, data, total_test_iterations,
                       speed_tester):
    # speed_tester.runtime_dict.update({'overall': []})

    with torch.no_grad():
        # init
        _ = model(return_loss=False, rescale=True, **data[0])
        for iteration in tqdm(range(total_test_iterations)):
            # start_time = timeit.default_timer()
            speed_tester.start_of_key('overall')
            _ = model(return_loss=False, rescale=True, **data[1])
            speed_tester.end_of_key('overall')

    average_time_overall = np.asarray(speed_tester.runtime_dict['overall']).mean()
    speed_tester.logger.info("The average overall runtime is: {} ms.".format(average_time_overall*1000))
    speed_tester.logger.info("The FPS is : {}".format(1 / average_time_overall))


def partial_speed_test(model, data, total_test_iterations,
                       speed_tester):
    # partial_keys = [
    #                 'backbone',
    #                 'neck',
    #                 'head0',
    #                 'head1',
    #                 'head2',
    #                 'head3',
    #                 'head4',
    #                 'head',
    #                 'post',
    #                 'nms'
    #                 ]
    # for key in partial_keys:
    #     speed_tester.runtime_dict.update({key: []})
    with torch.no_grad():
        # init
        _ = model(return_loss=False, rescale=True, **data[0])
        for iteration in tqdm(range(total_test_iterations)):
            _ = model(return_loss=False, rescale=True, **data[1])

    for key in speed_tester.runtime_dict.keys():
        _average_time_key = np.asarray(speed_tester.runtime_dict[key]).mean()
        speed_tester.logger.info("The average {} runtime is : {} ms.".format(key, _average_time_key*1000))
        # logger.info("The FPS is : {}".format(1000/average_time_overall))


if __name__ == '__main__':
    main()
