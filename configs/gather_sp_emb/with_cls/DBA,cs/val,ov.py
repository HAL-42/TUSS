#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/20 16:57
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.rslt_dir = ...

# -* 配置语义嵌入提取。
cfg.gather_cfg = Config(caps='configs/gather_sp_emb/clip,samq,DBA/cs_val.py')

@cfg.gather_cfg.set_IL(name='rslt_dir')  # noqa: E302
def gather_cfg_rslt_dir(c: Config):
    return c.rslt_dir

# -* 配置嵌入分类。
cfg.cls_cfg = Config(caps='configs/classify_sp/samq,clip/cs_val,ov.py')

@cfg.cls_cfg.set_IL(name='rslt_dir')  # noqa: E302
def cls_cfg_rslt_dir(c: Config):
    return str(Path(c.rslt_dir) / 'cls')

@cfg.cls_cfg.emb.set_IL(name='dir', priority=2)  # noqa: E302
def cls_cfg_emb_dir(c: Config):  # noqa: E302
    return Path(c.gather_cfg.rslt_dir) / 'emb'

# -* 分类配置无clip初始配置，而是使用gather提供的clip。
cfg.cls_cfg.clip.ini.empty_leaf()
