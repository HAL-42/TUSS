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

from alchemy_cat.py_tools import Config, IL

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

cfg.emb.dir = IL(lambda c: Path(c.rslt_dir) / '..' / 'emb')

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 80 / 100., 'norm': True},
                       {'softmax': 90 / 100., 'norm': True},
                       {'softmax': 100 / 100., 'norm': True},
                       {'softmax': 110 / 100., 'norm': True},
                       {'softmax': 120 / 100., 'norm': True}]
cfg.seed.bg_methods = [{'method': 'thresh', 'thresh': 0.2},
                       {'method': 'thresh', 'thresh': 0.3},
                       {'method': 'thresh', 'thresh': 0.4},
                       {'method': 'thresh', 'thresh': 0.5},
                       {'method': 'thresh', 'thresh': 0.6}]
