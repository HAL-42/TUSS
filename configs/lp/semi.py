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
from alchemy_cat.py_tools import Config, DEP

from configs.addon.semi_id_dic import cfg as semi_id_dic_cfg

cfg = config = Config('configs/lp/base.py')

# -* 配置数据包初始化。
cfg.lp_packet.semi_ids_key = '请指定semi_ids_key'
cfg.lp_packet.ini.semi_ids = DEP(lambda c: semi_id_dic_cfg[c.lp_packet.semi_ids_key], priority=0)
cfg.lp_packet.ini.emb_classify_rslt_dir = None

# -* 验证配置的种子点生成方式。
cfg.cls_cfg.cls_eval_dt.ini.val = True
cfg.cls_cfg.semi_ids = DEP(lambda c: c.lp_packet.ini.semi_ids)

cfg.cls_cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': True}]
