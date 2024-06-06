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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, ParamLazy

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/mask_clip/sof,th/emb')

cfg.pt.batch_size_factor_ = Param2Tune([1, 2, 4, 8, 16, 32, 64])
cfg.pt.lr_ = Param2Tune([1e-1, 1e-2])

cfg.loader.batch_size = ParamLazy(lambda c: c.pt.batch_size_factor_ * 256)

cfg.opt.get_pg.ini.lr = ParamLazy(lambda c: c.pt.lr_ * c.pt.batch_size_factor_)

cfg.sched.warm.warm_iters = ParamLazy(lambda c: 12_000 // c.pt.batch_size_factor_)
cfg.solver.max_iter = ParamLazy(lambda c: 120_000 // c.pt.batch_size_factor_)
