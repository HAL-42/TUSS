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

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/sam_auto_seg/c7/base.py',))

cfg.rslt_dir = ...

# * 设置训练、验证、测试集。
cfg.dt.ini.split = 'val'

# * 选择模型参数。
cfg.mask_gen.pattern_key = Param2Tune(['l2_nmsf_s1_rsw3',
                                       'l2_nmsf_s1_rsw3_64p',
                                       'l2_nmsf_s1_rsw3_128p',
                                       'ssa_heavy'])

# * 可视化参数。
cfg.viz.step = 100
