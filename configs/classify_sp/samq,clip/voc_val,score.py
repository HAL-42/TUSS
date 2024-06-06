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
from alchemy_cat.py_tools import Config

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

# -* 改为验证集。
cfg.seg_dt.ini.split = 'val'

# -* 数据输出改为验证模式。
cfg.cls_eval_dt.ini.val = True

# -* CLIP、文本分类器有关配置置空。
cfg.clip.empty_leaf()
cfg.txt_cls.empty_leaf()

# -* 配置通用分类器。
cfg.classifier.empty_leaf()
cfg.classifier.cls = "请配置score的分类器"

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': True},
                       {'softmax': None, 'norm': False, 'L1': False}]
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                       {'method': 'alpha_bg', 'alpha': 0.25},
                       {'method': 'alpha_bg', 'alpha': 0.5},
                       {'method': 'alpha_bg', 'alpha': 2.},
                       {'method': 'alpha_bg', 'alpha': 4.}]

cfg.seed.ini.with_bg_logit = True

cfg.seed.save_logit = True
