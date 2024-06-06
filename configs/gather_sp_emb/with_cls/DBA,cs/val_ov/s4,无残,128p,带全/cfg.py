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

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/DBA,cs/val,ov.py')

cfg.rslt_dir = ...

cfg.gather_cfg.extractor.ini.head_att_residual_weight = (0., 2.)

cfg.gather_cfg.extractor.ini.infer_method.stride_factor = 4
cfg.gather_cfg.extractor.ini.infer_method.with_global = True

cfg.gather_cfg.sam_sps.dir = Path('experiment/sam_auto_seg/cs/调多码,v/split=val,pattern_key=l2_nmsf_s1_rsw3_128p/anns')
