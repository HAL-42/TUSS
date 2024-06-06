#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/28 22:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config, Cfg2Tune, Param2Tune

from libs.label_propagation import LPClassifier

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/c14_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,c14/v/sli4,半全局,归一att') / 'emb'

@cfg.classifier.ini.set_DEP()
def packet(c: Config):
    return Path(c.rslt_dir).parents[1] / 'lp_classifier_ini' / 'ini.pkl'
cfg.classifier.ini.Y_star_method.method = 'seed_Y_star'
cfg.classifier.ini.k_bar = Param2Tune([12, 50, 25, 100])
cfg.classifier.ini.k = 50
cfg.classifier.ini.norm_a_method = Param2Tune(['rigorous', 'L1'])
cfg.classifier.cls = LPClassifier.samq_emb_classifier_from_lp_run
