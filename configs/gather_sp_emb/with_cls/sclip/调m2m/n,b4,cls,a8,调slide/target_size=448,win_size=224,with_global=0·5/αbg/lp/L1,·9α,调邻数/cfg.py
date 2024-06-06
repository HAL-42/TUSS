#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 14:31
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config, Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/lp/base.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
@cfg.lp_packet.ini.set_IL()
def emb_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[3] / 'emb'

@cfg.lp_packet.ini.set_IL()
def emb_classify_rslt_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[2] / 'emb_classify_rslt'

cfg.lp.ini.alpha = .9
cfg.lp.ini.gamma = 1
cfg.lp.ini.k = Param2Tune([400, 200, 100, 50, 25, 12])

cfg.cls_cfg.seed.fg_methods = [{'softmax': .5, 'norm': True, 'bypath_suppress': False},
                               {'softmax': .125, 'norm': True, 'bypath_suppress': False, 'L1': True},
                               {'softmax': .25, 'norm': True, 'bypath_suppress': False, 'L1': True},
                               {'softmax': .5, 'norm': True, 'bypath_suppress': False, 'L1': True},
                               {'softmax': 1., 'norm': True, 'bypath_suppress': False, 'L1': True},
                               {'softmax': 2., 'norm': True, 'bypath_suppress': False, 'L1': True}
                               ]
cfg.cls_cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 0.25},
                               {'method': 'alpha_bg', 'alpha': 0.5},
                               {'method': 'alpha_bg', 'alpha': 1.},
                               {'method': 'alpha_bg', 'alpha': 2.},
                               {'method': 'alpha_bg', 'alpha': 4.},
                               {'method': 'alpha_bg', 'alpha': 8.}]
