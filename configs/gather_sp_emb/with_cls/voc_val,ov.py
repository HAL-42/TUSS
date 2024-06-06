#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/20 16:57
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

from configs.addon.voc_names.clip_es import cfg as clip_es_name_cfg

cfg = config = Config()

cfg.rslt_dir = ...

# -* 配置语义嵌入提取。
cfg.gather_cfg = Config(caps='configs/gather_sp_emb/clip,samq,m2m_dba/slide.py')

@cfg.gather_cfg.set_IL(name='rslt_dir')  # noqa: E302
def gather_cfg_rslt_dir(c: Config):
    return c.rslt_dir

cfg.gather_cfg.dt.ini.split = 'val'
cfg.gather_cfg.sam_sps.dir = Path('experiment/sam_auto_seg/voc/split=val/anns')

# -* 配置嵌入分类。
cfg.cls_cfg = Config(cfgs_update_at_parser='configs/classify_sp/samq,clip/base.py')

@cfg.cls_cfg.set_IL(name='rslt_dir')  # noqa: E302
def cls_cfg_rslt_dir(c: Config):
    return str(Path(c.rslt_dir) / 'cls')

@cfg.cls_cfg.emb.set_IL(name='dir', priority=2)  # noqa: E302
def cls_cfg_emb_dir(c: Config):  # noqa: E302
    return Path(c.gather_cfg.rslt_dir) / 'emb'

# -* 改为验证集。
cfg.cls_cfg.seg_dt.ini.split = 'val'

# -* 数据输出改为验证模式。
cfg.cls_cfg.cls_eval_dt.ini.val = True

# -* 分类配置无clip初始配置，而是使用gather提供的clip。
cfg.cls_cfg.clip.ini.empty_leaf()

# -* OV只使用预定义类别名。
cfg.cls_cfg.txt_cls.ini.vocabulary = clip_es_name_cfg.fg_names

# -* 配置种子生成参数。
cfg.cls_cfg.seed.fg_methods = [{'softmax': 1., 'norm': False}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'pow', 'pow': 1.},  # 0.5
                               {'method': 'pow', 'pow': 2.},  # 0.382
                               {'method': 'pow', 'pow': 3.},  # 0.317
                               {'method': 'pow', 'pow': 5.},  # 0.245
                               {'method': 'pow', 'pow': .5},  # 0.618
                               {'method': 'pow', 'pow': .3},  # 0.698
                               {'method': 'pow', 'pow': .2}  # 0.755
                               ]
