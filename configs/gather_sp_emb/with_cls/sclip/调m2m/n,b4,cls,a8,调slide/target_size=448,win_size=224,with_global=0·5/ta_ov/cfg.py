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

cfg = config = Config(caps='configs/classify_sp/samq,clip/voc_val,ov.py')

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5') / 'emb'

cfg.seg_dt.ini.split = 'train_aug'
