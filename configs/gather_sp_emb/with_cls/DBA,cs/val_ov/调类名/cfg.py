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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

from configs.addon.cs_names import cfg as cs_name_cfg


cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/DBA,cs/val,ov.py')

cfg.rslt_dir = ...

cfg.cls_cfg.txt_cls.ini.vocabulary = Param2Tune([cs_name_cfg.ori_names, cs_name_cfg.semivl_names],
                                                optional_value_names=['ori', 'semivl'])
