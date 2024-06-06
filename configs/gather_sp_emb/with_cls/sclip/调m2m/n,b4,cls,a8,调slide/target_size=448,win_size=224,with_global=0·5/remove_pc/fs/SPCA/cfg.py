#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/29 20:26
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/cluster_metrics.py', caps='configs/feature_preprocess/remove_pc/base.py')

cfg.rslt_dir = ...

# -* 配置数据包初始化。
cfg.remove_pc_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                                        'target_size=448,win_size=224,with_global=0·5') / 'emb'
cfg.remove_pc_packet.ini.emb_classify_rslt_dir = None

# -* 配置PCA方法预处理。
cfg.pca.ini.sample_num = Param2Tune([100, 400, 1600, 6400, 25600])
cfg.pca.ini.resample_limit = 2

# -* 配置消除多少个主成分。
cfg.uut.remove_num = Param2Tune([1, 4, 8, 16])
