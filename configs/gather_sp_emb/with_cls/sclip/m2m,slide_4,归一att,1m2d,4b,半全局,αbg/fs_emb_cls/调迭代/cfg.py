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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, ParamLazy, Config

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

@cfg.dt.ini.set_DEP()
def root(c: Config):
    return Path(c.rslt_dir).parents[2] / 'emb'

cfg.pt.iter_factor_ = Param2Tune([8, 4, 2, 1, .5, .25])

cfg.sched.warm.warm_iters = ParamLazy(lambda c: int(6_000 // c.pt.iter_factor_))
cfg.solver.max_iter = ParamLazy(lambda c: int(60_000 // c.pt.iter_factor_))

@cfg.solver.set_DEP()
def save_step(c: Config):
    return max(c.solver.max_iter // 1, 1000)

cfg.val.cfg.seed.fg_methods = [{'softmax': 1., 'norm': True}]
cfg.val.cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                               {'method': 'alpha_bg', 'alpha': 0.25},
                               {'method': 'alpha_bg', 'alpha': 0.5},
                               {'method': 'alpha_bg', 'alpha': 2.},
                               {'method': 'alpha_bg', 'alpha': 4.}]
