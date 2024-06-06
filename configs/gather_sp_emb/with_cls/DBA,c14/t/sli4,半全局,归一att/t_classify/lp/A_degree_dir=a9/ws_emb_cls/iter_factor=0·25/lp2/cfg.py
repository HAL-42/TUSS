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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

from libs.label_propagation import LabelPropagator

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/lp/c14,base.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
cfg.lp_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att') / 'emb'
cfg.lp_packet.ini.emb_classify_rslt_dir = Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att/t_classify/lp/A_degree_dir=a9/ws_emb_cls/iter_factor=0·25/val/iter-final') / 'emb_classify_rslt'

# -* 配置lp。
cfg.lp.ini.empty_leaf()
cfg.lp.ini.A_degree_dir = Param2Tune([Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att/lp,A_degree/k50/alpha=0·7/A_degree'),
                                      Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att/lp,A_degree/k50/alpha=0·8/A_degree'),
                                      Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att/lp,A_degree/k50/alpha=0·9/A_degree'),
                                      Path('experiment/gather_sp_emb/with_cls/DBA,c14/t/sli4,半全局,归一att/lp,A_degree/k50/alpha=0·95/A_degree')],
                                     optional_value_names=['a7', 'a8', 'a9', 'a95'])
cfg.lp.ini.cupy = True
cfg.lp.cls = LabelPropagator.from_A_degree
