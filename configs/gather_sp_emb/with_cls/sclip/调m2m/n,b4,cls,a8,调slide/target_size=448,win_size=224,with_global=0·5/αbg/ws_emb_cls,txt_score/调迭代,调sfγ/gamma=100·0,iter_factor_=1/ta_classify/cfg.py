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

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5/emb')

cfg.seed.fg_methods = [{'softmax': 1., 'norm': True, 'bypath_suppress': True},
                       {'softmax': 1., 'norm': True, 'bypath_suppress': False},
                       {'softmax': .1, 'norm': True, 'bypath_suppress': False},
                       {'softmax': .25, 'norm': True, 'bypath_suppress': False},
                       {'softmax': .5, 'norm': True, 'bypath_suppress': False},
                       {'softmax': .75, 'norm': True, 'bypath_suppress': False},
                       {'softmax': 1.25, 'norm': True, 'bypath_suppress': False},
                       {'softmax': 1.5, 'norm': True, 'bypath_suppress': False}]
