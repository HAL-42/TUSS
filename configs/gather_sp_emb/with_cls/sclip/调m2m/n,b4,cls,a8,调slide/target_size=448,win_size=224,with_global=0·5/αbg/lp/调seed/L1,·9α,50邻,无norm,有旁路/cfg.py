#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 14:31
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/lp/base.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
@cfg.lp_packet.ini.set_IL()
def emb_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[3] / 'emb'

@cfg.lp_packet.ini.set_IL()
def emb_classify_rslt_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[2] / 'emb_classify_rslt'

cfg.cls_cfg.seed.fg_methods = [{'softmax': None, 'norm': True, 'bypath_suppress': False, 'L1': True},
                               {'softmax': None, 'norm': False, 'bypath_suppress': False, 'L1': True},
                               {'softmax': None, 'norm': True, 'bypath_suppress': True, 'L1': True}]
