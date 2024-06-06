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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/DBA,cs/val,ov.py')

cfg.rslt_dir = ...

cfg.gather_cfg.sam_sps.dir = Param2Tune([Path('experiment/sam_auto_seg/cs/调多码,v/split=val,pattern_key=l2_nmsf_s1_rsw3/anns'),
                                         Path('experiment/sam_auto_seg/cs/调多码,v/split=val,pattern_key=l2_nmsf_s1_rsw3_64p/anns'),
                                         Path('experiment/sam_auto_seg/cs/调多码,v/split=val,pattern_key=l2_nmsf_s1_rsw3_128p/anns'),
                                         Path('experiment/sam_auto_seg/cs/调多码,v/split=val,pattern_key=l2_nmsf_s1_rsw3_2l/anns'),
                                         Path('experiment/sam_auto_seg/cs/调多码,v/split=val,pattern_key=l2_nmsf_s1_rsw3_df1/anns'),
                                         Path('experiment/sam_auto_seg/cs/调多码,无rsw/split=val,pattern_key=ssa_heavy/anns'),
                                         Path('experiment/sam_auto_seg/cs/调多码,无rsw/split=val,pattern_key=ssa_heavy_128/anns'),
                                         ], optional_value_names=['rsw', 'rsw_64p', 'rsw_128p',
                                                                  'rsw_2l', 'rsw_df1', 'ssa_heavy', 'ssa_heavy_128'])
