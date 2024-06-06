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

cfg = config = Config(caps='configs/classify_sp/samq,clip/c14_val,probe.py')

cfg.rslt_dir = ...

cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,c14/v/sli4,半全局,归一att') / 'emb'

# -* 模型ckp路径。
@cfg.classifier.ini.set_DEP()
def ckp_pth(c: Config) -> Path:
    return Path(c.rslt_dir).parent / 'checkpoints' / 'last.pth'
