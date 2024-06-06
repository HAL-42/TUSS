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

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/voc_val,probe.py')

cfg.rslt_dir = ...

cfg.gather_cfg.extractor.ini.norm_att = True
cfg.gather_cfg.extractor.ini.head_att_residual_weight = (1., 2.)

# -* 模型ckp路径。
@cfg.cls_cfg.classifier.ini.set_DEP()
def ckp_pth(c: Config) -> Path:
    return Path(c.rslt_dir).parent / 'checkpoints' / 'last.pth'

cfg.cls_cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 2.},
                               {'method': 'alpha_bg', 'alpha': 4.},
                               {'method': 'alpha_bg', 'alpha': 8.}]
