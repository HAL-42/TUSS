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
from libs.data.coco2014 import COCO2014

cfg = config = Config(caps='configs/classify_sp/samq,clip/base.py')

# -* 改变数据集。
cfg.seg_dt.ini.empty_leaf()
cfg.seg_dt.ini.root = 'datasets'
cfg.seg_dt.ini.split = 'train'
cfg.seg_dt.ini.cls_labels_type = 'online'
cfg.seg_dt.ini.cocostuff_img = True
cfg.seg_dt.ini.cocostuff_lb = True
cfg.seg_dt.cls = COCO2014

# -* Obj最佳类别名。
cfg.txt_cls.ini.vocabulary = obj_name_cfg.sclip_fg_names + obj_name_cfg.es_bg_names

# -* 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': True}]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': 1.},  # 0.5
                       {'method': 'pow', 'pow': 2.},  # 0.382
                       {'method': 'pow', 'pow': 3.},  # 0.317
                       {'method': 'pow', 'pow': 5.},  # 0.245
                       {'method': 'pow', 'pow': .5},  # 0.618
                       {'method': 'pow', 'pow': .3},  # 0.698
                       {'method': 'pow', 'pow': .2}  # 0.755
                       ]
