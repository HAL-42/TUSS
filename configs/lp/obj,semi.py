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

cfg = config = Config('configs/lp/semi.py')

# -* 设定验证配置。
cfg.cls_cfg.empty_leaf()
cfg.cls_cfg = Config(caps='configs/classify_sp/samq,clip/base,obj.py')

# -** 设定验证配置的路径。
cfg.cls_cfg.rslt_dir = DEP(lambda c: c.rslt_dir)
cfg.cls_cfg.emb.dir = DEP(lambda c: c.lp_packet.ini.emb_dir)

# -* 设定数据。
cfg.cls_cfg.cls_eval_dt.ini.val = True
cfg.cls_cfg.semi_ids = DEP(lambda c: c.lp_packet.ini.semi_ids)

# -** 验证配置不构造分类器。
cfg.cls_cfg.clip.empty_leaf()
cfg.cls_cfg.txt_cls.empty_leaf()

# -* 验证配置的种子点生成方式。
cfg.cls_cfg.seed.fg_methods = [{'softmax': None, 'norm': False, 'L1': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                               {'method': 'alpha_bg', 'alpha': 0.25},
                               {'method': 'alpha_bg', 'alpha': 0.5},
                               {'method': 'alpha_bg', 'alpha': 2.},
                               {'method': 'alpha_bg', 'alpha': 4.}]
cfg.cls_cfg.seed.ini.with_bg_logit = True
