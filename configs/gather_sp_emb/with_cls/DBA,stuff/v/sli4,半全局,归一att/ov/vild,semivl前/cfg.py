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

import numpy as np
from alchemy_cat.py_tools import Config

from configs.addon.stuff_names import cfg as stuff_name_cfg

np.seterr(divide='ignore')  # 计算分类F1时不要警告。

cfg = config = Config('configs/addon/emb_cls_metrics.py',
                      caps='configs/classify_sp/samq,clip/stuff_val,ov,sclip前.py')

cfg.rslt_dir = ...

cfg.txt_cls.ini.vocabulary = stuff_name_cfg.semivl_names

@cfg.emb.set_IL(name='dir')  # noqa: E302
def emb_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[1] / 'emb'
