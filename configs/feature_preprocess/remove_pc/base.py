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

from alchemy_cat.py_tools import Config, DEP

from libs.emb_analysis.pca import SamplingPCA

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# -* 配置数据包初始化。
cfg.remove_pc_packet.ini.emb_dir = Path('请填写emb路径')
cfg.remove_pc_packet.ini.emb_classify_rslt_dir = Path('请填写emb_classify_rslt路径')

# -* 配置PCA方法预处理。
cfg.pca.ini.sample_num = -1
cfg.pca.ini.resample_limit = 2
cfg.pca.ini.rand_seed = DEP(lambda c: c.rand_seed)
cfg.pca.cls = SamplingPCA

# -* 配置消除多少个主成分。
cfg.uut.remove_num = 1
