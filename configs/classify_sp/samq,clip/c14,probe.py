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

from libs.data.coco2014 import COCO2014

cfg = config = Config(caps='configs/classify_sp/samq,clip/probe.py')

# -* 改为验证集。
cfg.seg_dt.ini.empty_leaf()
cfg.seg_dt.ini.root = 'datasets'
cfg.seg_dt.ini.split = 'train'
cfg.seg_dt.ini.cls_labels_type = 'online'
cfg.seg_dt.ini.cocostuff_img = True
cfg.seg_dt.ini.cocostuff_lb = True
cfg.seg_dt.cls = COCO2014
