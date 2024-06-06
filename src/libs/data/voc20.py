#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/12 3:32
@File    : voc20.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from addict import Dict
from alchemy_cat.contrib.voc import VOCAug2, VOC_COLOR

__all__ = ['kLabel2Color', 'VOC20']

kLabel2Color = np.full((256, 3), 255, dtype=np.uint8)
kLabel2Color[:20, :] = VOC_COLOR[1:, :]


class VOC20(VOCAug2):
    class_names = VOCAug2.class_names[1:]
    class_num = len(class_names)

    def get_item(self, index: int) -> Dict:
        out = super().get_item(index)

        out.cls_lb = out.cls_lb[1:]

        new_lb = out.lb.copy()
        new_lb[new_lb == 0] = 255
        new_lb[new_lb != 255] = new_lb[new_lb != 255] - 1

        out.lb = new_lb

        return out

    @staticmethod
    def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
        return kLabel2Color[label_map]
