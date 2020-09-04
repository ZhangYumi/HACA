from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def parse_args():
  
    parser = argparse.ArgumentParser(description='Train a detector')

    #读取config配置文件
    parser.add_argument('--config',
                        default='configs/faster_rcnn_r101_fpn_1x(without_non_local).py',
                        help='train config file path')

    #日志和模型保存路径
    parser.add_argument('--work_dir', help='the dir to save logs and models')

    #从断点加载参数训练
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')

    #训练期间验证，action='store_false'表示默认为True，如果有这个参数为False
    #反之，action='store_true'表示默认为False，如果有这个参数为True
    parser.add_argument(
        '--validate',
        action='store_true',        # action='store_true' means default=False when --validate is True.
        help='whether to evaluate the checkpoint during training')

    #gpu数量，默认设置为2
    parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')

    #随机种子
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    #启动器，只应用于分布式训练
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    #根据gpu数目自动调整学习率，此处默认为True，即action='store_false'
    parser.add_argument(
        '--autoscale_lr',
        action='store_false',        # action='store_false' means default=True when --autoscale_lr is None.
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args



def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
