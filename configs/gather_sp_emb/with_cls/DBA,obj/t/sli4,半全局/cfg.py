#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 20:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/clip,samq,DBA/base,obj.py')

cfg.rslt_dir = ...
