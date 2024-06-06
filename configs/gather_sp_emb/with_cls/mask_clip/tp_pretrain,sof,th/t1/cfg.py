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
from alchemy_cat.py_tools import Config, Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/addon/emb_cls_metrics.py', caps='configs/gather_sp_emb/with_cls/base.py')

cfg.rslt_dir = ...

# -* 配置语义嵌入提取。
cfg.gather_cfg = Cfg2Tune(caps='configs/gather_sp_emb/clip,samq,mask_clip/base.py')
cfg.gather_cfg.set_whole()

@cfg.gather_cfg.set_IL(name='rslt_dir')  # noqa: E302
def gather_cfg_rslt_dir(c: Config):
    return c.rslt_dir

clip_ini_cfgs = [  # noqa: E305
    Config({'model_name': 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 'pretrained': None}),
    Config({'model_name': 'hf-hub:apple/DFN5B-CLIP-ViT-H-14', 'pretrained': None}),
    # Config({'model_name': 'EVA02-E-14-plus', 'pretrained': 'laion2b_s9b_b144k'}),
    # Config({'model_name': 'ViT-SO400M-14-SigLIP-384', 'pretrained': 'webli'}),
    # Config({'model_name': 'ViT-bigG-14-CLIPA-336', 'pretrained': 'datacomp1b'}),
    # Config({'model_name': 'ViT-bigG-14-CLIPA', 'pretrained': 'datacomp1b'}),
    Config({'model_name': 'ViT-L-14-336', 'pretrained': 'openai'}),
    Config({'model_name': 'ViT-L-14', 'pretrained': 'openai'}),
    Config({'model_name': 'ViT-L-14', 'pretrained': 'commonpool_xl_clip_s13b_b90k'}),
    # Config({'model_name': 'ViT-bigG-14', 'pretrained': 'laion2b_s39b_b160k'}),
    Config({'model_name': 'ViT-L-14', 'pretrained': 'datacomp_xl_s13b_b90k'}),
]
clip_ini_names = [f'{ini_cfg.model_name.split("/")[-1]}--{ini_cfg.pretrained}'.replace('-', '_')
                  for ini_cfg in clip_ini_cfgs]

cfg.gather_cfg.clip.ini = Param2Tune(clip_ini_cfgs, optional_value_names=clip_ini_names)

# -* 配置种子生成参数。
cfg.cls_cfg.seed.fg_methods = [{'softmax': 1., 'norm': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'thresh', 'thresh': 0.1},
                               {'method': 'thresh', 'thresh': 0.2},
                               {'method': 'thresh', 'thresh': 0.3},
                               {'method': 'thresh', 'thresh': 0.4},
                               {'method': 'thresh', 'thresh': 0.5},
                               {'method': 'thresh', 'thresh': 0.6},
                               {'method': 'thresh', 'thresh': 0.7}]
