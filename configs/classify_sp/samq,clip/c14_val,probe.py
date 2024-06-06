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

cfg = config = Config(caps='configs/classify_sp/samq,clip/c14,probe.py')

# -* 改为验证集。
cfg.seg_dt.ini.split = 'val'

# -* 数据输出改为验证模式。
cfg.cls_eval_dt.ini.val = True

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': False}]
