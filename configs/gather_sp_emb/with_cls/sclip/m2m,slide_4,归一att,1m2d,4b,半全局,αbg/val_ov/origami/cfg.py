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

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,ov.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/m2m,slide_4,归一att,1m2d,4b,半全局,αbg/val_ov/emb')

cfg.txt_cls.templates.key = 'origami'

cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5},
                       {'method': 'pow', 'pow': .3},
                       {'method': 'pow', 'pow': .2},
                       {'method': 'pow', 'pow': .1},
                       {'method': 'pow', 'pow': .05}
                       ]
