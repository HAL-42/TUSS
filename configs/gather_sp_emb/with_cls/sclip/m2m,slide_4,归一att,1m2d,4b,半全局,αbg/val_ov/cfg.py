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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/voc_val,ov.py')

cfg.rslt_dir = ...

cfg.gather_cfg.extractor.ini.norm_att = True
cfg.gather_cfg.extractor.ini.head_att_residual_weight = (1., 2.)
