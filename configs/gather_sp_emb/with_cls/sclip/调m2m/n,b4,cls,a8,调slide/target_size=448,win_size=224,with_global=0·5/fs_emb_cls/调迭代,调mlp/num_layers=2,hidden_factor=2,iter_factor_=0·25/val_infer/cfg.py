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

from libs.classifier.emb_classifier import EmbMLP

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/voc_val,probe.py')

cfg.rslt_dir = ...

# -* 模型ckp路径。
@cfg.cls_cfg.classifier.ini.set_DEP()
def ckp_pth(c: Config) -> Path:
    return Path(c.rslt_dir).parent / 'checkpoints' / 'last.pth'

cfg.cls_cfg.classifier.ini.emb_dim = 512
cfg.cls_cfg.classifier.ini.cls_num = 21
cfg.cls_cfg.classifier.ini.scale = 100.
cfg.cls_cfg.classifier.ini.bias = False
cfg.cls_cfg.classifier.ini.cos_sim = True
cfg.cls_cfg.classifier.ini.num_layers = 2
cfg.cls_cfg.classifier.ini.hidden_factor = 2
cfg.cls_cfg.classifier.cls = EmbMLP.classify_samq_emb_val_classifier
