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
from pathlib import Path

from alchemy_cat.py_tools import Config

from libs.classifier.emb_classifier import EmbLinearProbe

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

# -* 改为验证集。
cfg.seg_dt.ini.split = 'val'

# -* 数据输出改为验证模式。
cfg.cls_eval_dt.ini.val = True

# -* CLIP、文本分类器有关配置置空。
cfg.clip.empty_leaf()
cfg.txt_cls.empty_leaf()

# -* 配置通用分类器。
cfg.classifier.ini.ckp_pth = Path('请填写模型ckp路径')
cfg.classifier.ini.emb_dim = 512
cfg.classifier.ini.cls_num = 21
cfg.classifier.ini.scale = 100.
cfg.classifier.ini.bias = False
cfg.classifier.ini.cos_sim = True
cfg.classifier.cls = EmbLinearProbe.classify_samq_emb_val_classifier

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': False}]
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                       {'method': 'alpha_bg', 'alpha': 0.25},
                       {'method': 'alpha_bg', 'alpha': 0.5},
                       {'method': 'alpha_bg', 'alpha': 2.},
                       {'method': 'alpha_bg', 'alpha': 4.}]

cfg.seed.ini.with_bg_logit = True
