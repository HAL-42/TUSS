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

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/lp/semi.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
cfg.lp_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                                 'target_size=448,win_size=224,with_global=0·5') / 'emb'
cfg.lp_packet.ini.emb_classify_rslt_dir = None
cfg.lp_packet.ini.semi_ids = Path('datasets/VOC2012/third_party/semi_split/pascal/662')

# -* 配置软标签Y预处理。
cfg.Y_method.method = 'bg_fg_balance'
cfg.Y_method.balance_ratio = 'eq'

# -* 设定标签传播器。
cfg.lp.ini.gamma = Param2Tune([1., 2., 3.])
cfg.lp.ini.alpha = Param2Tune([.7, .8, .9, .95])
cfg.lp.ini.k = Param2Tune([12, 25, 50, 100])
