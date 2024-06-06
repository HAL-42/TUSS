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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/gather_sp_emb/with_cls/base.py')

cfg.gather_cfg.extractor.ini.mask2bias.fg_bias = 0.
cfg.gather_cfg.extractor.ini.mask2bias.bg_bias = -8.

# -* 配置种子生成参数。
cfg.cls_cfg.seed.fg_methods = [{'softmax': 60 / 100., 'norm': True},
                               {'softmax': 80 / 100., 'norm': True},
                               {'softmax': 100 / 100., 'norm': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'thresh', 'thresh': 0.4},
                               {'method': 'thresh', 'thresh': 0.5},
                               {'method': 'thresh', 'thresh': 0.6},
                               {'method': 'thresh', 'thresh': 0.7},
                               {'method': 'thresh', 'thresh': 0.8}]
