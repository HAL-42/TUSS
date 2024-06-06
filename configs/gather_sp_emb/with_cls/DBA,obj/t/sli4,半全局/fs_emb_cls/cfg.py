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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/obj,base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/DBA,obj/t/sli4,半全局') / 'emb'
cfg.val.cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,obj/v/sli4,半全局') / 'emb'

cfg.solver.iter_factor = Param2Tune([2, 1, .5, .25, .125, .0625])
