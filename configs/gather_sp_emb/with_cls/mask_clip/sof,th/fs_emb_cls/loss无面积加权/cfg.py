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

from alchemy_cat.py_tools import Config

cfg = config = Config(caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/mask_clip/sof,th/emb')

cfg.loader.batch_size = 512

cfg.opt.get_pg.ini.lr = 0.02

cfg.sched.warm.warm_iters = 6_000
cfg.solver.max_iter = 60_000

cfg.loss.loss_items.classify.ini.sample_weighted = False
