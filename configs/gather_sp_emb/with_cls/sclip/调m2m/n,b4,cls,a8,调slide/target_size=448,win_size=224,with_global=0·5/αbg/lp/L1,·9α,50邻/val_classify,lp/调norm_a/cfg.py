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

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5') / 'val_ov' / 'emb'

@cfg.classifier.ini.set_DEP()
def packet(c: Config):
    return Path(c.rslt_dir).parents[2] / 'lp_classifier_ini' / 'ini.pkl'
cfg.classifier.ini.Y_star_method.method = Param2Tune(['seed_Y_star', 'raw_Y_star', 'raw_Y_star_suppress_non_fg'])
cfg.classifier.ini.k_bar = 50
cfg.classifier.ini.norm_a_method = Param2Tune(['rigorous', 'L1'])
cfg.classifier.cls = LPClassifier.samq_emb_classifier_from_lp_run

cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': True},
                       {'softmax': None, 'norm': False, 'L1': False}]
