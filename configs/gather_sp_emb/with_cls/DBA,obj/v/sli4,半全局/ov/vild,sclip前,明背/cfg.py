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

from configs.addon.obj_names import cfg as obj_name_cfg

cfg = config = Config('configs/addon/emb_cls_metrics.py',
                      caps='configs/classify_sp/samq,clip/obj_val,ov,origami,es背.py')

cfg.rslt_dir = ...

cfg.txt_cls.templates.key = 'vild'

cfg.txt_cls.ini.vocabulary = ['background'] + obj_name_cfg.sclip_fg_names

@cfg.emb.set_IL(name='dir')  # noqa: E302
def emb_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[1] / 'emb'

cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 0.125},
                       {'method': 'alpha_bg', 'alpha': 0.125 / 2},
                       {'method': 'alpha_bg', 'alpha': 0.125 / 4},
                       {'method': 'alpha_bg', 'alpha': 0.125 / 8}]

cfg.seed.ini.with_bg_logit = True
