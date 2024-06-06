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
from alchemy_cat.py_tools import Config

cfg = config = Config(cfgs_update_at_parser='configs/gather_sp_emb/with_cls/base.py')

cfg.rslt_dir = ...

# -* 配置语义嵌入提取。
cfg.gather_cfg = Config(cfgs_update_at_parser='configs/gather_sp_emb/clip,samq,sclip/base.py')
cfg.gather_cfg.set_whole()

@cfg.gather_cfg.set_IL(name='rslt_dir')  # noqa: E302
def gather_cfg_rslt_dir(c: Config):
    return c.rslt_dir

# -* 配置种子生成参数。
cfg.cls_cfg.seed.fg_methods = [{'softmax': None, 'norm': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'pow', 'pow': .05},
                               {'method': 'pow', 'pow': .1},
                               {'method': 'pow', 'pow': .15},
                               {'method': 'pow', 'pow': .2},
                               {'method': 'pow', 'pow': .25},
                               {'method': 'pow', 'pow': .3}]
