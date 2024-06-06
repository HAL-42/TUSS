#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/23 下午2:29
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/lp/base.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
cfg.lp_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/DBA,obj/t/sli4,半全局,归一att') / 'emb'
cfg.lp_packet.ini.emb_classify_rslt_dir = None

# -* 设定标签传播器。
cfg.lp.ini.alpha = Param2Tune([.7, .8, .9, .95])
cfg.lp.ini.k = 50
cfg.lp.ini.A_degree_only = True
