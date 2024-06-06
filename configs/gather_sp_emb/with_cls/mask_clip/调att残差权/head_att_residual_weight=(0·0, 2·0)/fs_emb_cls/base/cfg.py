#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/11 20:26
@File    : cfg.py
@Software: PyCharm
@Desc    :
"""
from pathlib import Path

from alchemy_cat.py_tools import Config, DEP

cfg = config = Config(caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = DEP(lambda c: Path(c.rslt_dir).parents[1] / 'emb')
