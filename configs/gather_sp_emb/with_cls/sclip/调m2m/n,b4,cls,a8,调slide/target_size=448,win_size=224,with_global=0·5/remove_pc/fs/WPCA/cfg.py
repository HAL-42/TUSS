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

from alchemy_cat.py_tools import Config

from libs.emb_analysis.pca import WeightedPCA

cfg = config = Config('configs/feature_preprocess/remove_pc/base.py')

# -* 配置数据包初始化。
cfg.remove_pc_packet.ini.emb_dir = Path('experiment/gather_sp_emb/with_cls/sclip/调m2m/n,b4,cls,a8,调slide/'
                                        'target_size=448,win_size=224,with_global=0·5') / 'emb'
cfg.remove_pc_packet.ini.emb_classify_rslt_dir = None

# -* 配置PCA方法预处理。
cfg.pca.ini.empty_leaf()
cfg.pca.cls = WeightedPCA

# -* 配置消除多少个主成分。
cfg.uut.remove_num = 1
