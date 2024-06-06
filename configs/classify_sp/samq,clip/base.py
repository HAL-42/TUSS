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
from pathlib import Path

from alchemy_cat.contrib.voc import VOCAug2
from alchemy_cat.py_tools import Config, IL

from configs.addon.clip_templates import cfg as clip_templates_cfg
from configs.addon.voc_names.clip_es import cfg as clip_es_name_cfg
from libs.clip.classifier import PredefinedOvClassifier
from libs.seeding.seed_anns import seed_on_logits

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# * 设定数据集。
cfg.seg_dt.ini.root = 'datasets'
cfg.seg_dt.ini.cls_labels_type = 'seg_cls_labels'
cfg.seg_dt.ini.split = 'train_aug'
cfg.seg_dt.cls = VOCAug2

# * 设定emb文件路径。
cfg.emb.dir = Path('请填入emb文件路径')

# * 设定骨干CLIP。
cfg.clip.ini.model_name = 'ViT-B-16'
cfg.clip.ini.pretrained = 'openai'

# * 设定基于文本的分类器。
cfg.txt_cls.templates.key = 'vild'
cfg.txt_cls.ini.cache_feature = True
cfg.txt_cls.ini.templates = IL(lambda c: clip_templates_cfg[c.txt_cls.templates.key])
cfg.txt_cls.ini.vocabulary = clip_es_name_cfg.fg_names + clip_es_name_cfg.bg_names
cfg.txt_cls.ini.dataset_name = None
cfg.txt_cls.cls = PredefinedOvClassifier.samq_emb_classifier

# * 配置种子生成参数。
cfg.seed.fg_methods = [{'softmax': 1., 'norm': True},
                       {'softmax': None, 'norm': True},
                       {'softmax': 1., 'norm': False}, ]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5},
                       {'method': 'pow', 'pow': .6},
                       {'method': 'pow', 'pow': .7},
                       {'method': 'pow', 'pow': .8},
                       {'method': 'pow', 'pow': .9},
                       {'method': 'pow', 'pow': 1},
                       {'method': 'pow', 'pow': 2},
                       {'method': 'pow', 'pow': 3}]

cfg.seed.ini.priority = ('level_bigger', 'conf_bigger')
cfg.seed.ini.ret_seeded_sps = True
cfg.seed.func = seed_on_logits

cfg.seed.save_logit = False

# * 配置本内调参。
cfg.tune.at = 'fg_bg_methods'
