#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 14:31
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config, Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/lp/base.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
@cfg.lp_packet.ini.set_IL()
def emb_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[3] / 'emb'

@cfg.lp_packet.ini.set_IL()
def emb_classify_rslt_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[2] / 'emb_classify_rslt'

cfg.lp.ini.alpha = Param2Tune([.9, .5, .7, .8, .85, .95])
cfg.lp.ini.k = Param2Tune([50, 12, 25, 100, 200])
