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

cfg = config = Config(caps='configs/classify_sp/samq,clip/voc_val,probe.py')

cfg.seg_dt.ini.split = 'train_aug'
cfg.cls_eval_dt.ini.val = False

# -* 模型ckp路径。
@cfg.classifier.ini.set_DEP()
def ckp_pth(c: Config) -> Path:
    return Path(c.rslt_dir).parent / 'checkpoints' / 'last.pth'

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/m2m,slide_4,归一att,1m2d,4b,半全局,αbg/emb')

cfg.seed.fg_methods = [{'softmax': 1., 'norm': True, 'bypath_suppress': False},
                       {'softmax': 1., 'norm': True}]
