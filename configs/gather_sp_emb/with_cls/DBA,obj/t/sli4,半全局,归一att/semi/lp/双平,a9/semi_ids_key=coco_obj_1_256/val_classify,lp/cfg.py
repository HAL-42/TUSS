#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/23 下午10:58
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

from libs.label_propagation import LPClassifier

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/obj_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,obj/v/sli4,半全局,归一att') / 'emb'

@cfg.classifier.ini.set_DEP()  # noqa: E302
def packet(c: Config):
    return Path(c.rslt_dir).parent / 'lp_classifier_ini' / 'ini.pkl'
cfg.classifier.ini.Y_star_method.method = 'seed_Y_star'  # noqa: E305
cfg.classifier.ini.k_bar = 50
cfg.classifier.ini.norm_a_method = 'rigorous'
cfg.classifier.ini.k = 50
cfg.classifier.cls = LPClassifier.samq_emb_classifier_from_lp_run
