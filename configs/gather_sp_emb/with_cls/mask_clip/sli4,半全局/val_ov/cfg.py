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

from libs.extractor.clip_extractor import MaskCLIP

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/DBA/voc_val,ov.py')

cfg.rslt_dir = ...

cfg.gather_cfg.extractor.empty_leaf()
cfg.gather_cfg.extractor.ini.head_layer_idx = -1
cfg.gather_cfg.extractor.cls = MaskCLIP

cfg.gather_cfg.extractor.ini.infer_method.method = 'slide'
cfg.gather_cfg.extractor.ini.infer_method.win_size = 224

cfg.gather_cfg.extractor.infer_stride_factor = 4
@cfg.gather_cfg.extractor.ini.infer_method.set_IL()  # noqa: E302
def stride(c: Config):
    return c.gather_cfg.extractor.ini.infer_method.win_size // c.gather_cfg.extractor.infer_stride_factor

cfg.gather_cfg.extractor.ini.infer_method.with_global = .5
