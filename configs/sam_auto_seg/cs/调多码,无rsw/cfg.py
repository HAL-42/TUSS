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

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/sam_auto_seg/cs/base.py',))

cfg.rslt_dir = ...

# * 设置训练、验证、测试集。
cfg.dt.ini.split = Param2Tune(['val', 'train'])

# * 选择模型参数。
cfg.mask_gen.pattern_key = Param2Tune(['ssa_heavy',
                                       'ssa_heavy_128'])

# * 可视化参数。
cfg.viz.step = 25
