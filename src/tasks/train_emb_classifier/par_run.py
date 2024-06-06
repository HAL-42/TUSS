#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 21:15
@File    : par_run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from alchemy_cat.py_tools import Config
from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.par_run.gather_metric import read_seg_cls_metric
from utils.argparse_exts import recover_cli_from_args
import tasks.train_emb_classifier.run as train_emb_classifier_run

# -* 读取命令行参数。
parser = train_emb_classifier_run.get_parser()
parser.add_argument('--purge', default=0, type=int)
args = parser.parse_args()

# -* 建立参数调优器。
runner = Cfg2TuneRunner(args.config,
                        config_root='configs',
                        experiment_root='experiment',
                        pool_size=torch.cuda.device_count())


# -* 定义参数调优过程。
@runner.register_work_fn
def work(pkl_idx, _, cfg_pkl, cfg_rslt_dir):
    # -* 根据分到的配置，运行任务。
    seg_metric_file = Path(cfg_rslt_dir) / 'val' / 'iter-final' / 'eval' / 'seg' / 'statistics.pkl'
    if (not args.purge) and seg_metric_file.is_file():
        print(f"{seg_metric_file}存在，跳过{cfg_pkl}。")
    else:
        # -* 找到当前应当使用的CUDA设备，并等待当前CUDA设备空闲。
        _, env_with_current_cuda = allocate_cuda_by_group_rank(pkl_idx, 1, block=False, verbosity=True)

        # -* 配置run脚本的args。
        run_args = argparse.Namespace(**vars(args))
        del run_args.purge
        run_args.config = cfg_pkl

        # -* 在当前设备上执行训练。
        subprocess.run([sys.executable, 'src/tasks/train_emb_classifier/run.py'] + recover_cli_from_args(run_args),
                       check=False, env=env_with_current_cuda)


# -* 定义参数调优过程中的数据收集过程。
@runner.register_gather_metric_fn
def gather_metric(_: Config, cfg_rslt_dir: str, __: ..., ___: dict[str, tuple[..., str]]) -> dict[str, ...]:
    return read_seg_cls_metric(Path(cfg_rslt_dir) / 'val' / 'iter-final' / 'eval')


runner.tuning()
