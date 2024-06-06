#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/11 22:21
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config, IL

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

cfg.emb.dir = IL(lambda c: Path(c.rslt_dir).parent / 'emb')

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': True}]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': 1.},  # 0.5
                       {'method': 'pow', 'pow': 2.},  # 0.382
                       {'method': 'pow', 'pow': 1.794},  # 0.4
                       {'method': 'pow', 'pow': 3.},  # 0.317
                       {'method': 'pow', 'pow': 7.},  # 0.203
                       {'method': 'pow', 'pow': .5},  # 0.618
                       {'method': 'pow', 'pow': .3},  # 0.698
                       ]
