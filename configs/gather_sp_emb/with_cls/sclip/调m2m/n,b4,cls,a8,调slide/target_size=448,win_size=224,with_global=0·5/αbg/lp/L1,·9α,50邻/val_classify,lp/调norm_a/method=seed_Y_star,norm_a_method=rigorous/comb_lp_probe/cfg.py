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
import operator
from pathlib import Path

from alchemy_cat.py_tools import Param2Tune, Cfg2Tune

from libs.classifier.aux_classifier import LinearCombineClassifier, GatherClassifierResultReader

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5') / 'val_ov' / 'emb'

cfg.classifier.ini.score_suppress.eps = 1e-8
cfg.classifier.ini.score_suppress.comb_op = operator.or_
cfg.classifier.ini.lp.ini.emb_dir = cfg.emb.dir  # FIXME
cfg.classifier.ini.lp.ini.emb_classify_rslt_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/target_size=448,win_size=224,with_global=0·5/αbg/lp/L1,·9α,50邻/val_classify,lp/调norm_a/method=seed_Y_star,norm_a_method=rigorous/emb_classify_rslt')
cfg.classifier.ini.lp.cls = GatherClassifierResultReader
cfg.classifier.ini.lp.comb_weight = Param2Tune([.9, .5, 1.5, 2., 2.5, 3.])
cfg.classifier.ini.probe.ini.emb_dir = cfg.emb.dir  # FIXME
cfg.classifier.ini.probe.ini.emb_classify_rslt_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/target_size=448,win_size=224,with_global=0·5/αbg/lp/L1,·9α,调邻数/k=50/ws_emb_cls,txt_score/调迭代/iter_factor_=1/val_classify/emb_classify_rslt')
cfg.classifier.ini.probe.cls = GatherClassifierResultReader
cfg.classifier.ini.probe.comb_weight = 1.
cfg.classifier.cls = LinearCombineClassifier

cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': True},
                       {'softmax': None, 'norm': False, 'L1': False}]
