#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/13 14:58
@File    : cls_lb.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch

__all__ = ['lb2cls_lb', 'cls_lb2fg_cls_lb', 'cls_lb2fg_cls']


def lb2cls_lb(label: np.ndarray, class_num: int, ignore_label: int=255) -> np.ndarray:
    return (np.bincount(label.ravel(), minlength=ignore_label + 1) != 0).astype(np.uint8)[:class_num]


def cls_lb2fg_cls_lb(cls_lb: torch.Tensor) -> torch.Tensor:
    """Convert class label to foreground class label.

    Args:
        cls_lb: [..., K] class label tensor.

    Returns:
        [N, K-1] foreground class label tensor.
    """
    return cls_lb[..., 1:]


def cls_lb2fg_cls(cls_lb: torch.Tensor) -> torch.Tensor:
    """Convert class label to foreground class label.

    Args:
        cls_lb: [K,] class label tensor.

    Returns:
        [F,] foreground class label tensor.
    """
    return torch.nonzero(cls_lb2fg_cls_lb(cls_lb), as_tuple=True)[0]
