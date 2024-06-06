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
from alchemy_cat.py_tools import Config

from configs.addon.voc_names.clip_es import cfg as clip_es_name_cfg
from libs.clip.visual import FeatureExtractor, SemWithMaskPooling

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
cfg.sam_sps.dir = Path('experiment/sam_auto_seg/voc/split=train_aug/anns')

# * 设定骨干CLIP。
cfg.clip.ini.model_name = 'ViT-B-16'
cfg.clip.ini.pretrained = 'openai'

# * 设定特征提取器。
cfg.extractor.ini.last_layer_idx = -1
cfg.extractor.cls = FeatureExtractor

# * 设定语义聚合器。
cfg.sem_agg.cls = SemWithMaskPooling
cfg.sem_agg.fg_names = clip_es_name_cfg.fg_names
