#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/8 22:04
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config(cfgs_update_at_parser=('configs/sam_auto_seg/c7/base.py',))

cfg.rslt_dir = ...

# * 设置训练、验证、测试集。
cfg.dt.ini.split = 'train'

# * 可视化参数。
cfg.viz.step = 100
