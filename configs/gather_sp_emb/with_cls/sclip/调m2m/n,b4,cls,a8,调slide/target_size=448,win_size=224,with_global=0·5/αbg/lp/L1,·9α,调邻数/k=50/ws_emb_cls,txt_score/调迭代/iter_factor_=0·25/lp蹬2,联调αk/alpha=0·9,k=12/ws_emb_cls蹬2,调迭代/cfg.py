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

from libs.data import SAMQEmbDt

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                       'target_size=448,win_size=224,with_global=0·5') / 'emb'
@cfg.dt.ini.set_DEP()  # noqa: E302
def embwise_h5(c: Config):
    return Path(c.rslt_dir).parents[1] / 'emb_classify_rslt' / 'embwise.h5'
cfg.dt.ini.score2prob.norm = 'softmax'  # noqa: E305
cfg.dt.ini.score2prob.gamma = 100.
cfg.dt.cls = SAMQEmbDt.from_embwise_h5

cfg.pt.iter_factor_ = Param2Tune([4, 2, 1, .5, .25, .125])

cfg.sched.warm.warm_iters = ParamLazy(lambda c: int(6_000 // c.pt.iter_factor_))
cfg.solver.max_iter = ParamLazy(lambda c: int(60_000 // c.pt.iter_factor_))
