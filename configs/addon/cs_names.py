#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/10 22:50
@File    : cs_names.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()


cfg.ori_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

cfg.semivl_names = [','.join(p) for p in [
    ['road', 'street', 'parking space'],
    ['sidewalk'],
    ['building', 'skyscaper', 'house', 'bus stop building', 'garage', 'car port', 'scaffolding'],
    ['individual standing wall, which is not part of a building'],
    ['fence', 'hole in fence'],
    ['pole', 'sign pole', 'traffic light pole'],
    ['traffic light'],
    ['traffic sign', 'parking sign', 'direction sign'],
    ['vegetation', 'tree', 'hedge'],
    ['terrain', 'grass', 'soil', 'sand'],
    ['sky'],
    ['person', 'pedestrian', 'walking person', 'standing person', 'person sitting on the ground',
     'person sitting on a bench', 'person sitting on a chair'],
    ['rider', 'cyclist', 'motorcyclist'],
    ['car', 'jeep', 'SUV', 'van'],
    ['truck', 'box truck', 'pickup truck', 'truck trailer'],
    ['bus'],
    ['train', 'tram'],
    ['motorcycle', 'moped', 'scooter'],
    ['bicycle'],
]]
