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

from alchemy_cat.py_tools import Config, IL

from libs.label_propagation import LabelPropagator

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# -* 配置数据包初始化。
cfg.lp_packet.ini.emb_dir = Path('请填写emb路径')
cfg.lp_packet.ini.emb_classify_rslt_dir = Path('请填写emb_classify_rslt路径')

# -* 配置特征X预处理。
cfg.X_method.method = 'L2'

# -* 配置软标签Y预处理。
cfg.Y_method.method = 'identical'

# cfg.Y_method.method = 'area_conf_weighted'
# cfg.Y_method.alpha = .5
# cfg.Y_method.area.norm = prep.maxabs_scale  # 比例不变，保持物理意义——像素数目的大小。
# cfg.Y_method.area.gamma = .5
# cfg.Y_method.conf.cal = partial(np.amax, axis=-1)
# cfg.Y_method.conf.norm = prep.minmax_scale  # 消除最低置信度底噪。
# cfg.Y_method.conf.gamma = 1.

# -* 设定标签传播器。
cfg.lp.ini.k = 50
cfg.lp.ini.max_iter = 200  # NOTE 极大，几乎一定会收敛。实践中可配合增加tol、减小max_iter来提高速度。
cfg.lp.ini.alpha = .9
cfg.lp.ini.gamma = 1.
cfg.lp.ini.cg_guess = False
cfg.lp.cls = LabelPropagator
cfg.lp.cal = LabelPropagator.lp_packet_cal

# -* 设定验证配置。
cfg.cls_cfg = Config(caps='configs/classify_sp/samq,clip/base.py')

# -** 设定验证配置的路径。
cfg.cls_cfg.rslt_dir = IL(lambda c: c.rslt_dir)
cfg.cls_cfg.emb.dir = IL(lambda c: c.lp_packet.ini.emb_dir)

# -** 验证配置不构造分类器。
cfg.cls_cfg.clip.empty_leaf()
cfg.cls_cfg.txt_cls.empty_leaf()

# -* 验证配置的种子点生成方式。
cfg.cls_cfg.seed.fg_methods = [{'softmax': None, 'norm': True, 'bypath_suppress': False, 'L1': True},
                               {'softmax': .25, 'norm': True, 'bypath_suppress': False, 'L1': True, 'is_score': True},
                               {'softmax': .5, 'norm': True, 'bypath_suppress': False, 'L1': True, 'is_score': True},
                               {'softmax': 1., 'norm': True, 'bypath_suppress': False, 'L1': True, 'is_score': True}]
cfg.cls_cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1.},
                               {'method': 'alpha_bg', 'alpha': 0.25},
                               {'method': 'alpha_bg', 'alpha': 0.5},
                               {'method': 'alpha_bg', 'alpha': 2.},
                               {'method': 'alpha_bg', 'alpha': 4.}]
cfg.cls_cfg.seed.ini.with_bg_logit = True
cfg.cls_cfg.seed.ini.drop_bg_logit = False
