#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/23 0:53
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import typing as t

import numpy as np
import sklearn.preprocessing as prep
from scipy.stats import entropy

__all__ = ['complement_entropy', 'area_conf_weight']


def complement_entropy(prob: np.ndarray) -> np.ndarray:
    """Compute complement entropy with ( log2(cls_num) - entropy(prob) ) / log2(cls_num).

    Args:
        prob: (N, C, ...) prob. prob will be normalized if they don't sum to 1.

    Returns:
        (N, ...) complement entropy.
    """
    c = prob.shape[1]
    return 1. - entropy(prob, axis=1) / np.log(c)


def area_conf_weight(area: np.ndarray, score: np.ndarray,
                     alpha: float,
                     area_norm: t.Callable[[np.ndarray], np.ndarray], area_gamma: float,
                     conf_cal: t.Callable[[np.ndarray], np.ndarray], conf_norm: t.Callable[[np.ndarray], np.ndarray],
                     conf_gamma: float) -> np.ndarray:
    # -* 计算基于面积的权重。
    area_weight = area_norm(area) ** area_gamma  # [N,],f4

    # -* 计算基于置信度的权重。
    conf = conf_cal(score)  # [N,],f4
    conf_weight = conf_norm(conf) ** conf_gamma  # [N,],f4

    # -* 计算最终的权重。
    weight = alpha * area_weight + (1 - alpha) * conf_weight  # [N,],f4
    weight = prep.maxabs_scale(weight)  # NOTE 此归一化只会线性缩放Y*，稳定数值范围，不会改变Y的分布。为了加速可去掉。

    return weight
