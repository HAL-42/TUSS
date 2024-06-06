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
from libs.data.coco164k_dt import COCOObj

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

# -* 改为验证集。
cfg.seg_dt.ini.empty_leaf()
cfg.seg_dt.ini.root = 'datasets/cocostuff164k'
cfg.seg_dt.ini.split = 'train'
cfg.seg_dt.cls = COCOObj

# -* OV只使用预定义类别名。
cfg.txt_cls.ini.vocabulary = obj_name_cfg.es_fg_names + obj_name_cfg.es_bg_names
