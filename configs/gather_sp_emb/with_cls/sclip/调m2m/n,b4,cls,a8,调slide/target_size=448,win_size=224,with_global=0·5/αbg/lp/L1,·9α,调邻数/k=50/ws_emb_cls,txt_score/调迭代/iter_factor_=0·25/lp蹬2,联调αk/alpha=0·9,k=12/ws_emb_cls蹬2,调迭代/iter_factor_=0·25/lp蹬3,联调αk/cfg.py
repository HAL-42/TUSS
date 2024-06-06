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
cfg.lp_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                                 'target_size=448,win_size=224,with_global=0·5') / 'emb'

@cfg.lp_packet.ini.set_IL()
def emb_classify_rslt_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[1] / 'val' / 'iter-final' / 'emb_classify_rslt'

cfg.lp.ini.alpha = Param2Tune([.9, .7, .95])
cfg.lp.ini.k = Param2Tune([50, 12, 25])

cfg.cls_cfg.seed.fg_methods = [{'softmax': None, 'norm': True, 'bypath_suppress': False, 'L1': True}]  # 只做纯L1，减少调参量。
