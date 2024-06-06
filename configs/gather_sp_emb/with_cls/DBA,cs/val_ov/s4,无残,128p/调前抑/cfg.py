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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/cs_val,ov.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,cs/val_ov/s4,无残,128p/emb')

cfg.seed.fg_methods = Param2Tune([[{'softmax': 1., 'norm': False, 'fg_suppress': {'thresh': .1, 'keep_bg': False}}],
                                  [{'softmax': 1., 'norm': False, 'fg_suppress': {'thresh': .3, 'keep_bg': False}}],
                                  [{'softmax': 1., 'norm': False, 'fg_suppress': {'thresh': .5, 'keep_bg': False}}],
                                  [{'softmax': 1., 'norm': False, 'fg_suppress': {'thresh': .7, 'keep_bg': False}}]],
                                 optional_value_names=['t1', 't3', 't5', 't7'])
