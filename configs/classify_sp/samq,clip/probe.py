#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 14:31
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, DEP

from libs.classifier.emb_classifier import EmbLinearProbe

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

# -* CLIP、文本分类器有关配置置空。
cfg.clip.empty_leaf()
cfg.txt_cls.empty_leaf()

# -* 配置通用分类器。
cfg.classifier.ini.ckp_pth = ...
cfg.classifier.ini.emb_dim = 512
cfg.classifier.ini.cls_num = DEP(lambda c: c.seg_dt.cls.class_num)
cfg.classifier.ini.scale = 100.
cfg.classifier.ini.bias = False
cfg.classifier.ini.cos_sim = True
cfg.classifier.cls = EmbLinearProbe.classify_samq_emb_val_classifier

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': True, 'bypath_suppress': False}]
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                       {'method': 'alpha_bg', 'alpha': 0.25},
                       {'method': 'alpha_bg', 'alpha': 0.5},
                       {'method': 'alpha_bg', 'alpha': 2.},
                       {'method': 'alpha_bg', 'alpha': 4.}]

cfg.seed.ini.with_bg_logit = True
