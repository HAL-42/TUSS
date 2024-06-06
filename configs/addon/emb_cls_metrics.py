#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 22:15
@File    : emb_cls_metrics.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.metric_names = ['mIoU', 'seg_precision', 'seg_recall', 'seg_accuracy', 'seg_cls_IoU',
                    'cls_F1', 'cls_recall', 'cls_precision', 'cls_accuracy']
