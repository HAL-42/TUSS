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
from pathlib import Path

from alchemy_cat.py_tools import Config

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/classify_sp/samq,clip/cs_val,ov.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,cs/val_ov/s4,无残,128p/emb')

cfg.txt_cls.templates.key = 'origami'
