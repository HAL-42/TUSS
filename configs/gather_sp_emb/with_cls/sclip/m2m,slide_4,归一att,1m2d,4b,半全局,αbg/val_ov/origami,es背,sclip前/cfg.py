#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 20:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

from configs.addon.voc_names.clip_es import cfg as clip_es_name_cfg

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/voc_val,ov.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/m2m,slide_4,归一att,1m2d,4b,半全局,αbg/val_ov/emb')

cfg.txt_cls.templates.key = 'origami'

cfg.txt_cls.ini.vocabulary = clip_es_name_cfg.sclip_fg_names + clip_es_name_cfg.bg_names

cfg.seed.bg_methods = [{'method': 'pow', 'pow': 1.},  # 0.5
                       {'method': 'pow', 'pow': 2.},  # 0.382
                       {'method': 'pow', 'pow': 3.},  # 0.317
                       {'method': 'pow', 'pow': 5.},  # 0.245
                       {'method': 'pow', 'pow': .5},  # 0.618
                       {'method': 'pow', 'pow': .3},  # 0.698
                       {'method': 'pow', 'pow': .2}  # 0.755
                       ]
