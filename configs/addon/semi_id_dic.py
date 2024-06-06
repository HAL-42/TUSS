#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/24 下午1:35
@File    : semi_id_dic.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

cfg = Config()

cfg.voc_aug_1_16 = Path('datasets/VOC2012/third_party/semi_split/pascal/662')
cfg.voc_aug_1_8 = Path('datasets/VOC2012/third_party/semi_split/pascal/1323')
cfg.voc_aug_1_4 = Path('datasets/VOC2012/third_party/semi_split/pascal/2646')
cfg.voc_aug_1_2 = Path('datasets/VOC2012/third_party/semi_split/pascal/5291')

cfg.voc_aug_u2pl_1_16 = Path('datasets/VOC2012/third_party/semi_split/pascal_u2pl/662')
cfg.voc_aug_u2pl_1_8 = Path('datasets/VOC2012/third_party/semi_split/pascal_u2pl/1323')
cfg.voc_aug_u2pl_1_4 = Path('datasets/VOC2012/third_party/semi_split/pascal_u2pl/2646')
cfg.voc_aug_u2pl_1_2 = Path('datasets/VOC2012/third_party/semi_split/pascal_u2pl/5291')

cfg.coco_obj_1_512 = Path('datasets/cocostuff164k/third_party/semi_split/coco_obj/1_512')
cfg.coco_obj_1_256 = Path('datasets/cocostuff164k/third_party/semi_split/coco_obj/1_256')
cfg.coco_obj_1_128 = Path('datasets/cocostuff164k/third_party/semi_split/coco_obj/1_128')
cfg.coco_obj_1_64 = Path('datasets/cocostuff164k/third_party/semi_split/coco_obj/1_64')
cfg.coco_obj_1_32 = Path('datasets/cocostuff164k/third_party/semi_split/coco_obj/1_32')
