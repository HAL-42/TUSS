#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/21 20:41
@File    : par_run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os.path as osp
import subprocess
import sys

import torch
from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank
from rich.console import Console

sys.path = ['.', './src'] + sys.path  # noqa: E402

con = Console(color_system='standard')

parser = argparse.ArgumentParser()
parser.add_argument('--purge', default=0, type=int)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-b', '--points_per_batch', default=256, type=int)
parser.add_argument('-w', '--rock_sand_water_chunk_size', default=0, type=int)
parser.add_argument('-m', '--output_mode', default='uncompressed_rle', type=str)
args = parser.parse_args()

runner = Cfg2TuneRunner(args.config,
                        config_root='configs',
                        experiment_root='experiment',
                        pool_size=torch.cuda.device_count())


@runner.register_work_fn
def work(pkl_idx, _, cfg_pkl, cfg_rslt_dir):
    # * 根据分到的配置，运行任务。
    if (not args.purge) and osp.isdir(ann_save_dir := osp.join(cfg_rslt_dir, 'anns')):
        con.print(f"[bold red][跳过][/bold red] {ann_save_dir}存在，跳过{cfg_pkl}。")
    else:
        # * 找到当前应当使用的CUDA设备，并等待当前CUDA设备空闲。
        _, env_with_current_cuda = allocate_cuda_by_group_rank(pkl_idx, 1, block=True, verbosity=True)

        # * 在当前设备上执行训练。
        subprocess.run([sys.executable, 'src/tasks/sam_auto_seg/run.py',
                        '-b', f'{args.points_per_batch}',
                        '-w', f'{args.rock_sand_water_chunk_size}',
                        '-m', args.output_mode,
                        '-c', cfg_pkl],
                       check=False, env=env_with_current_cuda)


runner.tuning()
