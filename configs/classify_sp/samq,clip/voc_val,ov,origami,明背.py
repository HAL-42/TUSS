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

from configs.addon.voc_names.clip_es import cfg as clip_es_name_cfg

cfg = config = Config(caps='configs/classify_sp/samq,clip/voc_val,ov.py')

cfg.txt_cls.templates.key = 'origami'

cfg.txt_cls.ini.vocabulary = ['background'] + clip_es_name_cfg.fg_names

cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                       {'method': 'alpha_bg', 'alpha': 0.5},
                       {'method': 'alpha_bg', 'alpha': 0.25},
                       {'method': 'alpha_bg', 'alpha': 0.125},
                       {'method': 'alpha_bg', 'alpha': 0.125 / 2}]

cfg.seed.ini.with_bg_logit = True
