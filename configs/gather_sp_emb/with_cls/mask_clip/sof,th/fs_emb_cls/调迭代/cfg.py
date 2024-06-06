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

cfg.pt.iter_factor_ = Param2Tune([0.25, 0.5, 1, 2, 4, 8])

cfg.loader.batch_size = 512

cfg.opt.get_pg.ini.lr = 0.02

cfg.sched.warm.warm_iters = ParamLazy(lambda c: int(12_000 // c.pt.iter_factor_))
cfg.solver.max_iter = ParamLazy(lambda c: int(120_000 // c.pt.iter_factor_))
