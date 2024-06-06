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

from alchemy_cat.py_tools import Config

from libs.label_propagation import LPClassifier

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5') / 'test_ov' / 'emb'

cfg.seg_dt.ini.split = 'test'

cfg.classifier.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                                  'target_size=448,win_size=224,with_global=0·5') / 'emb'
cfg.classifier.ini.k = 50
cfg.classifier.ini.k_bar = None  # 与k随动。
cfg.classifier.ini.Y_star_method.method = 'seed_Y_star'
cfg.classifier.ini.norm_a_method = 'rigorous'
cfg.classifier.cls = LPClassifier.samq_emb_classifier_from_emb_fs

cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': False}]
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 2.}]
