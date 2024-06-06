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

cfg = config = Config(caps='configs/classify_sp/samq,clip/voc_val,probe.py')

# -* 模型ckp路径。
@cfg.classifier.ini.set_DEP()
def ckp_pth(c: Config) -> Path:
    return Path(c.rslt_dir).parent / 'checkpoints' / 'last.pth'

cfg.classifier.ini.emb_dim = 512
cfg.classifier.ini.cls_num = 21
cfg.classifier.ini.scale = 100.
cfg.classifier.ini.bias = False
cfg.classifier.ini.cos_sim = True
cfg.classifier.ini.num_layers = 2
cfg.classifier.ini.hidden_factor = 2
cfg.classifier.cls = EmbMLP.classify_samq_emb_val_classifier

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                   'target_size=448,win_size=224,with_global=0·5/fs_emb_cls/调迭代,调mlp/'
                   'num_layers=2,hidden_factor=2,iter_factor_=0·25/val_infer/emb')
