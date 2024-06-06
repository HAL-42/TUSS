#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/29 20:26
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

cfg = config = Config(caps='configs/train_emb_classifier/base.py')

# -* 设定调率器。
cfg.sched.warm.warm_iters = 12_000  # ~2 epoch
cfg.solver.max_iter = IL(lambda c: int(120_000 // c.solver.iter_factor), priority=0)  # ~20 epoch

# -* 设定验证配置。
cfg.val.empty_leaf()
cfg.val.cfg = Config(caps='configs/classify_sp/samq,clip/c14,probe.py')

# -** 设定验证配置的路径。
cfg.val.cfg.rslt_dir = '请填入验证结果保存路径'
cfg.val.cfg.emb.dir = IL(lambda c: c.dt.ini.root)

# -** 验证配置不构造分类器。
cfg.val.cfg.classifier.empty_leaf()
