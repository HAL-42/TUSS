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

from configs.addon.obj_names import cfg as obj_name_cfg

cfg = config = Config(caps='configs/classify_sp/samq,clip/obj_val,ov,origami,es背.py')

cfg.txt_cls.ini.vocabulary = ['background'] + obj_name_cfg.es_fg_names

cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                       {'method': 'alpha_bg', 'alpha': 0.5},
                       {'method': 'alpha_bg', 'alpha': 0.25},
                       {'method': 'alpha_bg', 'alpha': 0.125},
                       {'method': 'alpha_bg', 'alpha': 0.125 / 2}]

cfg.seed.ini.with_bg_logit = True
