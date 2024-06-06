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
from alchemy_cat.py_tools import Cfg2Tune, Config, Param2Tune

from libs.extractor.clip_extractor import MaskCLIP

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/base.py')

cfg.rslt_dir = ...

# -* 配置语义嵌入提取。
cfg.gather_cfg = Cfg2Tune(caps='configs/gather_sp_emb/clip,samq,mask_clip/base.py')
cfg.gather_cfg.set_whole()

@cfg.gather_cfg.set_IL(name='rslt_dir')  # noqa: E302
def gather_cfg_rslt_dir(c: Config):
    return c.rslt_dir

# -* 配置m2m。
cfg.gather_cfg.extractor.ini.head_att_bias_method.method = 'binary_accord_mask'  # noqa: E305
cfg.gather_cfg.extractor.ini.head_att_bias_method.downsample_method = Param2Tune(['max', 'nearest'])
cfg.gather_cfg.extractor.ini.head_att_bias_method.fg_bias = 0.
cfg.gather_cfg.extractor.ini.head_att_bias_method.bg_bias = Param2Tune([-8., -4., -2.])
cfg.gather_cfg.extractor.ini.head_att_bias_method.at = (True,)
cfg.gather_cfg.extractor.ini.head_att_bias_method.can_see_cls_tok = Param2Tune([True, False])
cfg.gather_cfg.extractor.ini.sclip_custom_att_factor = Param2Tune([2., 8., 32.])
cfg.gather_cfg.extractor.cls = MaskCLIP.sclip

# -* 配置种子生成参数。
cfg.cls_cfg.seed.fg_methods = [{'softmax': 1., 'norm': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'thresh', 'thresh': 0.2},
                               {'method': 'thresh', 'thresh': 0.3},
                               {'method': 'thresh', 'thresh': 0.4},
                               {'method': 'thresh', 'thresh': 0.5},
                               {'method': 'thresh', 'thresh': 0.6},
                               {'method': 'thresh', 'thresh': 0.7},
                               {'method': 'thresh', 'thresh': 0.8}]
