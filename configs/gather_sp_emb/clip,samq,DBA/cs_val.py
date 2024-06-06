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

from libs.data.cityscapes_dt import CityScapes
from libs.extractor.clip_extractor import MaskCLIP

cfg = config = Config(caps='configs/gather_sp_emb/clip,samq,att_pool/base.py')

# -* 设定数据集。
cfg.dt.ini.empty_leaf()
cfg.dt.ini.root = 'datasets/cityscapes'
cfg.dt.ini.split = 'val'
cfg.dt.cls = CityScapes

# -* 配置DBA。
cfg.extractor.set_whole()
cfg.extractor.ini.head_layer_idx = -1
cfg.extractor.ini.head_att_bias_method.method = 'binary_accord_mask'  # noqa: E305
cfg.extractor.ini.head_att_bias_method.downsample_method = 'nearest'
cfg.extractor.ini.head_att_bias_method.fg_bias = 0.
cfg.extractor.ini.head_att_bias_method.bg_bias = -4.
cfg.extractor.ini.head_att_bias_method.at = (True,)
cfg.extractor.ini.head_att_bias_method.can_see_cls_tok = False
cfg.extractor.ini.norm_att = False
cfg.extractor.ini.head_att_residual_weight = (1., 1.)

cfg.extractor.cls = MaskCLIP.sclip

# -* 配置滑窗。
cfg.auger.ini.scale_crop_method.set_whole()  # noqa: E305
cfg.auger.ini.scale_crop_method.method = 'fix_short_no_crop'
cfg.auger.ini.scale_crop_method.target_size = 512
# NOTE 短边缩放到的target_size总是patch倍数，理论上不会被aligner改变；而长边会对齐，以让图片既可以slide，又可以one-forward。
cfg.auger.ini.scale_crop_method.aligner = "请按照CLIP设置"

cfg.extractor.ini.infer_method.method = 'multi_slide'
cfg.extractor.ini.infer_method.stride_factor = 2
cfg.extractor.ini.infer_method.win_sizes = (448, 224)
cfg.extractor.ini.infer_method.with_global = False
cfg.extractor.ini.infer_method.momentum = 0.5
cfg.extractor.ini.infer_method.level_cumsum = False

# -* 配置pool方法。
cfg.extractor.ini.pool_method = 'up_emb_avg'

# -* 设定SAM superpixels数据位置。
cfg.sam_sps.dir = DEP(lambda c: Path(f'experiment/sam_auto_seg/cs/调多码,v/split={c.dt.ini.split},'
                                     f'pattern_key=l2_nmsf_s1_rsw3/anns'))
