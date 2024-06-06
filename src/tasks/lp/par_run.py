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
import os
import subprocess
import sys
from pathlib import Path

from alchemy_cat.py_tools import Config
from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.par_run.gather_metric import read_seg_cls_metric

# -* 读取命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str)
parser.add_argument('--purge', default=0, type=int)
parser.add_argument('-d', '--is_debug', default=0, type=int)
args = parser.parse_args()

# -* 建立参数调优器。
runner = Cfg2TuneRunner(args.config,
                        pool_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))


# -* 定义参数调优过程。
@runner.register_work_fn
def work(pkl_idx, _, cfg_pkl, cfg_rslt_dir):
    # -* 根据分到的配置，运行任务。
    seg_metric_file = Path(cfg_rslt_dir) / 'eval' / 'seg' / 'statistics.pkl'
    if (not args.purge) and seg_metric_file.is_file():
        print(f"{seg_metric_file}存在，跳过{cfg_pkl}。")
    else:
        # -* 找到当前应当使用的CUDA设备，并等待当前CUDA设备空闲。
        _, env_with_current_cuda = allocate_cuda_by_group_rank(pkl_idx, 1, block=True, verbosity=True)

        # -* 在当前设备上执行训练。
        subprocess.run([sys.executable, 'src/tasks/lp/run.py',
                        '-d', str(args.is_debug),
                        '-c', cfg_pkl],
                       check=False, env=env_with_current_cuda)


# -* 定义参数调优过程中的数据收集过程。
@runner.register_gather_metric_fn
def gather_metric(_: Config, cfg_rslt_dir: str, __: ..., ___: dict[str, tuple[..., str]]) \
        -> dict[str, ...]:
    return read_seg_cls_metric(Path(cfg_rslt_dir) / 'eval')


runner.tuning()
