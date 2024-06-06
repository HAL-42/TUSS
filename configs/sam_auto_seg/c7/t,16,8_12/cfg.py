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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

from libs.data.coco164k_dt import COCOStuff

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/sam_auto_seg/c7/base.py',))

cfg.rslt_dir = ...

# * 设置训练、验证、测试集。
cfg.dt.ini.split_idx_num = Param2Tune([(8, 16), (9, 16), (10, 16), (11, 16)])
cfg.dt.ini.split = 'train'
cfg.dt.cls = COCOStuff.subset

# * 可视化参数。
cfg.viz.step = 100
