#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 11:15
@File    : clip_es.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()


cfg.fg_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair seat', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                ]
cfg.bg_names = ['ground', 'land', 'grass', 'tree', 'building',
                'wall', 'sky', 'lake', 'water', 'river',
                'sea', 'railway', 'railroad', 'keyboard', 'helmet',
                'cloud', 'house', 'mountain', 'ocean', 'road',
                'rock', 'street', 'valley', 'bridge', 'sign',
                ]

cfg.sclip_fg_names = ['aeroplane',
                      'bicycle',
                      'bird',
                      'ship',
                      'bottle',
                      'bus',
                      'car',
                      'cat',
                      'chair',
                      'cow',
                      'table',
                      'dog',
                      'horse',
                      'motorbike',
                      'person, person in shirt, person in jeans, person in dress, '
                      'person in sweater, person in skirt, person in jacket',
                      'pottedplant',
                      'sheep',
                      'sofa',
                      'train',
                      'television monitor, tv monitor, monitor, television, screen']
