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

from alchemy_cat.py_tools import Param2Tune, Cfg2Tune, Config

from libs.classifier.aux_classifier import LinearCombineClassifier, GatherClassifierResultReader

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,score.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5') / 'val_ov' / 'emb'

cfg.classifier.ini.score_suppress.eps = 1e-8
cfg.classifier.ini.score_suppress.comb_op = operator.or_
cfg.classifier.cls = LinearCombineClassifier

@cfg.classifier.ini.lp.ini.set_DEP('emb_dir')  # noqa: E302
def lp_emb_dir(c: Config) -> Path:
    return c.emb.dir
@cfg.classifier.ini.lp.ini.set_DEP('emb_classify_rslt_dir')  # noqa: E302
def lp_emb_classify_rslt_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[1] / 'emb_classify_rslt'
cfg.classifier.ini.lp.cls = GatherClassifierResultReader  # noqa: E305

cfg.classifier.ini.probe.comb_weight = 1.
@cfg.classifier.ini.probe.ini.set_DEP('emb_dir')
def probe_emb_dir(c: Config) -> Path:  # noqa: E302
    return c.emb.dir
cfg.classifier.ini.probe.ini.emb_classify_rslt_dir = Param2Tune([Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/target_size=448,win_size=224,with_global=0·5/fs_emb_cls/调迭代/iter_factor_=0·5') / 'val_classify' / 'emb_classify_rslt',
                                                                 Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/target_size=448,win_size=224,with_global=0·5/fs_emb_cls/调迭代/iter_factor_=0·5') / 'val_classify,bg_alpha=2' / 'emb_classify_rslt',
                                                                 Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/target_size=448,win_size=224,with_global=0·5/fs_emb_cls/调迭代/iter_factor_=0·25') / 'val_classify' / 'emb_classify_rslt',
                                                                 Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/target_size=448,win_size=224,with_global=0·5/fs_emb_cls/调迭代/iter_factor_=0·25') / 'val_classify,bg_alpha=2' / 'emb_classify_rslt',],
                                                                optional_value_names=['i5b4', 'i5b2', 'i25b5', 'i25b2'])
cfg.classifier.ini.probe.cls = GatherClassifierResultReader

cfg.classifier.ini.lp.comb_weight = Param2Tune([.5, .9, 1.5, 2., 2.5, 3.])
