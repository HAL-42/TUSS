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

cfg.emb.dir = IL(lambda c: Path(c.rslt_dir) / '..' / 'emb')  # noqa: E305

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 60 / 100, 'norm': True}]
cfg.seed.bg_methods = [{'method': 'thresh', 'thresh': .8},
                       {'method': 'thresh', 'thresh': .85},
                       {'method': 'thresh', 'thresh': .9},
                       {'method': 'thresh', 'thresh': .95}]
