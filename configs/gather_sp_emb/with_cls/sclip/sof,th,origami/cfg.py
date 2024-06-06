#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 20:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 在configs/gather_sp_emb/with_cls/sclip/sof,th基础上，改为使用origami context。
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

# * 配置CLIP文本分类器模板。
cfg.cls_cfg.txt_cls.templates.key = 'origami'

# -* 配置种子生成参数。
cfg.cls_cfg.seed.fg_methods = [{'softmax': 65 / 100., 'norm': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'thresh', 'thresh': 0.35},
                               {'method': 'thresh', 'thresh': 0.4},
                               {'method': 'thresh', 'thresh': 0.45},
                               {'method': 'thresh', 'thresh': 0.5}]
