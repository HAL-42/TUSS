#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/28 20:43
@File    : typing.py
@Software: PyCharm
@Desc    : 
"""
import typing as t

import torch

__all__ = ['SPEmbClassifier']


SPEmbClassifier: t.TypeAlias = t.Callable[[torch.Tensor, str], torch.Tensor]
