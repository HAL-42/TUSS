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
import json
import subprocess
import sys
from pathlib import Path

from alchemy_cat.py_tools import Config
from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner

sys.path = ['.', './src'] + sys.path  # noqa: E402

# -* 读取命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('--purge', default=0, type=int)
parser.add_argument('--pool_size', default=0, type=int)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-d', '--is_debug', default=0, type=int)
args = parser.parse_args()

# -* 建立参数调优器。
runner = Cfg2TuneRunner(args.config, pool_size=args.pool_size)


# -* 定义参数调优过程。
@runner.register_work_fn
def work(_, __, cfg_pkl, cfg_rslt_dir):
    # -* 根据分到的配置，运行任务。
    metric_file = Path(cfg_rslt_dir) / 'eval' / 'statistics.json'
    if (not args.purge) and metric_file.is_file():
        print(f"{metric_file}存在，跳过{cfg_pkl}。")
    else:
        # -* 在当前设备上执行训练。
        subprocess.run([sys.executable, 'src/tasks/feature_preprocess/remove_pc/run.py',
                        '-d', str(args.is_debug),
                        '-c', cfg_pkl],
                       check=False)


# -* 定义参数调优过程中的数据收集过程。
@runner.register_gather_metric_fn
def gather_metric(_: Config, cfg_rslt_dir: str, __: ..., ___: dict[str, tuple[..., str]]) \
        -> dict[str, ...]:

    metrics = json.loads((Path(cfg_rslt_dir) / 'eval' / 'statistics.json').read_text())

    return metrics


runner.tuning()
