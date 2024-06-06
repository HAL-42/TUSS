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

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,ov,origami,es背.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/mask_clip/sli4,半全局/val_ov/emb')

cfg.seed.bg_methods = [{'method': 'pow', 'pow': 5.},  # 0.245
                       {'method': 'pow', 'pow': 7.},  # 0.618
                       {'method': 'pow', 'pow': 9.},  # 0.698
                       {'method': 'pow', 'pow': 11.},
                       {'method': 'pow', 'pow': 13.},
                       ]
