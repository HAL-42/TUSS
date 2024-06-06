#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/11 20:26
@File    : cfg.py
@Software: PyCharm
@Desc    :
"""
from pathlib import Path

from alchemy_cat.py_tools import Config

from configs.addon.semi_id_dic import cfg as semi_id_dic_cfg
from libs.data import SAMQEmbDt

cfg = config = Config('configs/addon/emb_cls_metrics.py', caps='configs/train_emb_classifier/obj,base.py')

cfg.rslt_dir = ...

cfg.val.cfg.emb.dir = Path('experiment/gather_sp_emb/with_cls/DBA,obj/v/sli4,半全局,归一att') / 'emb'

cfg.dt.ini.root = Path('experiment/gather_sp_emb/with_cls/DBA,obj/t/sli4,半全局,归一att') / 'emb'
@cfg.dt.ini.set_DEP()
def embwise_h5(c: Config):
    return Path(c.rslt_dir).parent / 'emb_classify_rslt' / 'embwise.h5'
cfg.dt.ini.prob2soft_lb.norm = 'softmax'  # noqa: E305
cfg.dt.ini.prob2soft_lb.gamma = 100.
cfg.dt.ini.semi_ids = semi_id_dic_cfg['coco_obj_1_32']
cfg.dt.ini.conf_method.method = 'no_conf'
cfg.dt.cls = SAMQEmbDt.from_embwise_h5

cfg.solver.iter_factor = .25
