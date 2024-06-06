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

from alchemy_cat.contrib.voc import VOCAug2, VOCAuger
from alchemy_cat.py_tools import Config, DEP

from libs.extractor.clip_extractor import AttPoolCLIPWithMaskLikeSAN

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# * 设定数据集。
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.ini.split = 'train_aug'
cfg.dt.cls = VOCAug2

# * 设定训练和测试数据增强器。
cfg.auger.ini.is_color_jitter = False
cfg.auger.ini.scale_crop_method.method = 'scale_align'
cfg.auger.ini.scale_crop_method.aligner = "请按照CLIP设置"
cfg.auger.ini.scale_crop_method.scale_factors = [1.]
cfg.auger.ini.is_rand_mirror = False
cfg.auger.ini.mean = "请按照CLIP配套的preprocess设置"
cfg.auger.ini.std = "请按照CLIP配套的preprocess设置"
cfg.auger.ini.lb_scale_factor = None
cfg.auger.ini.ol_cls_lb = False
cfg.auger.cls = VOCAuger

# * 设定SAM superpixels数据位置。
cfg.sam_sps.dir = DEP(lambda c: Path(f'experiment/sam_auto_seg/voc/split={c.dt.ini.split}/anns'))

# * 设定骨干CLIP。
cfg.clip.ini.model_name = 'ViT-B-16'
cfg.clip.ini.pretrained = 'openai'

# * 设定语义嵌入提取器。
cfg.extractor.ini.head_layer_idx = 9
cfg.extractor.ini.mask2bias.method = 'binary'
cfg.extractor.ini.mask2bias.fg_bias = 2.0
cfg.extractor.ini.mask2bias.bg_bias = -8.0
cfg.extractor.ini.bias_downsample_method = 'max'
cfg.extractor.cls = AttPoolCLIPWithMaskLikeSAN
