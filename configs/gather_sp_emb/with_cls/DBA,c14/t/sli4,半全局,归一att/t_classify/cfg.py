#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 20:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/base,c14.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att') / 'emb'

cfg.seed.fg_methods = [{'softmax': 1., 'norm': True},
                       {'softmax': 1., 'norm': False}]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': 1.},  # 0.5
                       {'method': 'pow', 'pow': .3},  # 0.698
                       {'method': 'pow', 'pow': .2}  # 0.755
                       ]
