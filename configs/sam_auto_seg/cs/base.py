#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/8 20:06
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

from configs.sam_auto_seg.patterns.mask_gen_ini import cfg as mask_gen_ini_cfgs
from libs.data.cityscapes_dt import CityScapes

cfg = config = Config(cfgs_update_at_parser=('configs/sam_auto_seg/base.py',))

cfg.rslt_dir = ...

# * 设置训练、验证、测试集。
cfg.dt.ini.empty_leaf()
cfg.dt.ini.root = 'datasets/cityscapes'
cfg.dt.ini.split = ...
cfg.dt.ini.rgb_img = True
cfg.dt.cls = CityScapes

# * 选择模型参数。
cfg.mask_gen.pattern_key = 'l2_nmsf_s1_rsw3'
@cfg.mask_gen.set_IL()  # noqa: E302
def ini(c):
    return mask_gen_ini_cfgs[c.mask_gen.pattern_key].branch_copy()
