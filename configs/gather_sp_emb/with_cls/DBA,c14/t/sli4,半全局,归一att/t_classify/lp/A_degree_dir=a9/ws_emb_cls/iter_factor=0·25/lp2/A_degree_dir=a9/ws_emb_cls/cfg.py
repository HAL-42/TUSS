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

from libs.data import SAMQEmbDt

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/c14,base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att') / 'emb'
cfg.dt.ini.embwise_h5 = Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att/t_classify/lp/A_degree_dir=a9/ws_emb_cls/iter_factor=0·25/lp2/A_degree_dir=a9') / 'emb_classify_rslt' / 'embwise.h5'
cfg.dt.ini.score2prob.norm = 'softmax'
cfg.dt.ini.score2prob.gamma = 100.
cfg.dt.cls = SAMQEmbDt.from_embwise_h5

cfg.solver.iter_factor = Param2Tune([1, .5, .25])
