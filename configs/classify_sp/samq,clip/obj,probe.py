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

from libs.data.coco164k_dt import COCOObj

cfg = config = Config(caps='configs/classify_sp/samq,clip/probe.py')

# -* 改为验证集。
cfg.seg_dt.ini.empty_leaf()
cfg.seg_dt.ini.root = 'datasets/cocostuff164k'
cfg.seg_dt.ini.split = 'train'
cfg.seg_dt.cls = COCOObj

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': True, 'bypath_suppress': False},
                       {'softmax': 1., 'norm': False, 'bypath_suppress': False}]
