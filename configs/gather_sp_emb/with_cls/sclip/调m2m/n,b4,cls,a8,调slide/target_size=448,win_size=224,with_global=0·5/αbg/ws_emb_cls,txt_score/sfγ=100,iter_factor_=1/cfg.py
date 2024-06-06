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

from libs.data import SAMQEmbDt

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

@cfg.dt.ini.set_DEP()
def root(c: Config):
    return Path(c.rslt_dir).parents[2] / 'emb'
@cfg.dt.ini.set_DEP()
def embwise_h5(c: Config):
    return Path(c.rslt_dir).parents[1] / 'emb_classify_rslt' / 'embwise.h5'
cfg.dt.ini.score2prob.norm = 'softmax'
cfg.dt.ini.score2prob.gamma = 100.
cfg.dt.cls = SAMQEmbDt.from_embwise_h5

cfg.pt.iter_factor_ = 1

cfg.sched.warm.warm_iters = DEP(lambda c: int(6_000 // c.pt.iter_factor_), priority=-1)
cfg.solver.max_iter = DEP(lambda c: int(60_000 // c.pt.iter_factor_), priority=-1)

@cfg.solver.set_DEP()
def save_step(c: Config):
    return max(c.solver.max_iter // 1, 1000)

cfg.val.cfg.seed.fg_methods = [{'softmax': 1., 'norm': True},
                               {'softmax': 1., 'norm': False}]
cfg.val.cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                               {'method': 'alpha_bg', 'alpha': 0.25},
                               {'method': 'alpha_bg', 'alpha': 0.5},
                               {'method': 'alpha_bg', 'alpha': 2.},
                               {'method': 'alpha_bg', 'alpha': 4.}]
