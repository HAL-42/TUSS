#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 17:24
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import sys
from pprint import pp

import torch
from alchemy_cat.py_tools import Config, ADict
from alchemy_cat.torch_tools import init_env

sys.path = ['.', './src'] + sys.path  # noqa: E402

from tasks.gather_clip_by_samq import run2 as gather_clip_by_samq_run
from tasks.classify_samq_emb import run as classify_samq_emb_run


def main(args: argparse.Namespace, cfg: str | Config) -> ADict:
    # -* 获取配置。
    gather_cfg = cfg.gather_cfg.unfreeze().parse(experiments_root='').compute_item_lazy().freeze()
    cls_cfg = cfg.cls_cfg.unfreeze().parse(experiments_root='').compute_item_lazy().freeze()
    pp(gather_cfg)
    pp(cls_cfg)

    # -* 提取clip语义嵌入。
    gather_ret = gather_clip_by_samq_run.main(args, gather_cfg)

    # -* 释放显存。
    torch.cuda.empty_cache()

    # -* 基于clip语义嵌入的分类。
    cls_rslt = classify_samq_emb_run.main(args, cls_cfg, gather_ret.open_clip_created)

    return cls_rslt


if __name__ == '__main__':
    # -* 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-b", '--benchmark', default=0, type=int)
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    _args = parser.parse_args()

    # -* 初始化环境。
    _, _cfg = init_env(is_cuda=True,
                       is_benchmark=bool(_args.benchmark),
                       is_train=False,
                       config_path=_args.config,
                       experiments_root="experiment",
                       rand_seed=0,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=bool(_args.is_debug))

    main(_args, _cfg)
