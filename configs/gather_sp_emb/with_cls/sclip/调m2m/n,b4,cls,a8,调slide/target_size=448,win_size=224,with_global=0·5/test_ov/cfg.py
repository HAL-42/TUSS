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

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/clip,samq,m2m_dba/slide.py')

cfg.rslt_dir = ...

cfg.dt.ini.split = 'test'
cfg.sam_sps.dir = Path('experiment/sam_auto_seg/voc/split=test/anns')
