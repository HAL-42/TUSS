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

from alchemy_cat.py_tools import Config, Cfg2Tune, Param2Tune

from libs.label_propagation import LPClassifier

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5') / 'val_ov' / 'emb'

@cfg.classifier.ini.set_DEP()  # noqa: E302
def packet(c: Config):
    return Path(c.rslt_dir).parents[1] / 'lp_classifier_ini' / 'ini.pkl'
cfg.classifier.ini.Y_star_method.method = 'seed_Y_star'  # noqa: E305
cfg.classifier.ini.Y_star_method.raw_Y_star_suppress_non_fg = True
cfg.classifier.ini.k_bar = Param2Tune([(25, 25), (15, 35), (35, 15), (10, 40), (5, 45),
                                       (50, 50), (33, 67), (67, 33), (20, 80), (10, 90),
                                       (12, 13), (8, 17), (17, 8)])
cfg.classifier.ini.norm_a_method = Param2Tune(['rigorous', 'L1'])
cfg.classifier.cls = LPClassifier.samq_emb_classifier_from_lp_run
