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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py',
                        cfgs_update_at_parser='configs/gather_sp_emb/with_cls/base.py')

cfg.rslt_dir = ...

# -* 配置att_pool的静态偏置。
cfg.gather_cfg.extractor.ini.mask2bias.fg_bias = Param2Tune([0., 2.])
cfg.gather_cfg.extractor.ini.mask2bias.bg_bias = Param2Tune([-4., -8., -12.])

# -* 配置本内调参。
cfg.cls_cfg.seed.fg_methods = [{'softmax': 1., 'norm': True},
                               {'softmax': 1., 'norm': False},]
cfg.cls_cfg.seed.bg_methods = [{'method': 'pow', 'pow': .0025},
                               {'method': 'pow', 'pow': .005},
                               {'method': 'pow', 'pow': .01},
                               {'method': 'pow', 'pow': .015}]
