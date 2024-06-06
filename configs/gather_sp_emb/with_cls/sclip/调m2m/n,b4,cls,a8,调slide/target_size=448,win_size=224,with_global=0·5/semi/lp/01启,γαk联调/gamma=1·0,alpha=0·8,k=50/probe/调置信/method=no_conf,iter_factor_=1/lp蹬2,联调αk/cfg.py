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

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/lp/semi.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
cfg.lp_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                                 'target_size=448,win_size=224,with_global=0·5') / 'emb'
@cfg.lp_packet.ini.set_IL()
def emb_classify_rslt_dir(c: Config) -> Path:
    return Path(c.rslt_dir).parents[1] / 'val' / 'iter-final' / 'emb_classify_rslt'
cfg.lp_packet.semi_ids_key = 'voc_aug_1_16'

# -* 配置软标签Y预处理。
cfg.Y_method.method = Param2Tune(['bg_fg_balance', 'identical'])
cfg.Y_method.balance_ratio = 'eq'

# -* 配置LP参数。
cfg.lp.ini.alpha = Param2Tune([.1, .2, .4, .5, .6, .7, .8, .9,])
cfg.lp.ini.k = Param2Tune([6, 12, 25, 50, 100])
