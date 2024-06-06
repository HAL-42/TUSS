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

cfg = config = Config(caps='configs/classify_sp/samq,clip/c14_val,probe.py')

# -* 配置通用分类器。
cfg.classifier.empty_leaf()
cfg.classifier.cls = "请配置score的分类器"

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': True}]
