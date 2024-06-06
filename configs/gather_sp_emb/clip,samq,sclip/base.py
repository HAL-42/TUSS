#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/29 20:26
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

from libs.extractor.clip_extractor import SCLIP

cfg = config = Config(cfgs_update_at_parser='configs/gather_sp_emb/clip,samq,att_pool/base.py')

# * 设定语义嵌入提取器。
cfg.extractor.set_whole()
cfg.extractor.ini.head_layer_idx = -1
cfg.extractor.cls = SCLIP
