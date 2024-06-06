#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/11 20:26
@File    : cfg.py
@Software: PyCharm
@Desc    :
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

cfg = config = Config(caps='configs/train_emb_classifier/base.py')

cfg.rslt_dir = ...

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/mask_clip/调slide/t1/'
                       'target_size=448,win_size=224,infer_stride_factor=4,with_global=0.25/emb')
