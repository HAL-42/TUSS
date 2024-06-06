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

from configs.addon.cs_names import cfg as cs_name_cfg
from libs.data.cityscapes_dt import CityScapes

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

# -* 改为验证集。
cfg.seg_dt.ini.empty_leaf()
cfg.seg_dt.ini.root = 'datasets/cityscapes'
cfg.seg_dt.ini.split = 'val'
cfg.seg_dt.cls = CityScapes

# -* 数据输出改为验证模式。
cfg.cls_eval_dt.ini.val = True

# -* OV只使用预定义类别名。
cfg.txt_cls.ini.vocabulary = cs_name_cfg.ori_names

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': False}]
cfg.seed.bg_methods = [{'method': 'no_bg'}]
cfg.seed.ini.with_bg_logit = True
