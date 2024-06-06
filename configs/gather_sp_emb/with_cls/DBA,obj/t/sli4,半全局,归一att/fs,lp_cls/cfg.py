#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/16 下午4:20
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

from libs.label_propagation import LPClassifier

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/obj_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,obj/v/sli4,半全局,归一att') / 'emb'

cfg.classifier.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/DBA,obj/t/sli4,半全局,归一att') / 'emb'
cfg.classifier.ini.k = Param2Tune([12, 25, 50, 100])
cfg.classifier.ini.k_bar = None
cfg.classifier.ini.Y_star_method.method = 'seed_Y_star'
cfg.classifier.ini.norm_a_method = Param2Tune(['rigorous', 'L1'])
# cfg.classifier.ini.suppress_no_fg_gather = Param2Tune([True, False])  # TODO
cfg.classifier.cls = LPClassifier.samq_emb_classifier_from_emb_fs

cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                       {'method': 'alpha_bg', 'alpha': 2.},
                       {'method': 'alpha_bg', 'alpha': 4.},
                       {'method': 'alpha_bg', 'alpha': 6.}]
