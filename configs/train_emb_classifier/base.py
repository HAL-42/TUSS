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

import torch.optim.lr_scheduler as sched
from alchemy_cat.py_tools import Config, IL
from torch import optim
from torch.utils.data import RandomSampler

from libs.classifier.emb_classifier import EmbLinearProbe
from libs.data import SAMQEmbDt
from libs.loss.emb_classify_loss import EmbClassifyLoss

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# -* 设定随机性。
cfg.rand.data = 0
cfg.rand.model = 0
cfg.rand.train = 0

# -* 设定数据集。
cfg.dt.ini.root = Path('请填入gather_sp_emb实验的emb目录')
cfg.dt.cls = SAMQEmbDt

# -* 设定训练和测试数据增强器。
@cfg.auger.set_func()  # noqa: E302
def cls(dt, *_, **__):  # 原样返回原始dt。
    return dt

# -* 设定数据加载器。
cfg.sampler.cls = RandomSampler  # noqa: E305

cfg.loader.batch_size = 512
cfg.loader.num_workers = 16

# -* 设定模型。
cfg.model.ini.scale = 100.
cfg.model.ini.bias = False
cfg.model.ini.cos_sim = True
cfg.model.cls = EmbLinearProbe
cfg.model.cal = EmbLinearProbe.emb_dt_inp_cal
cfg.model.val_cal = EmbLinearProbe.classify_samq_emb_cal

# -* 设定loss函数。
cfg.loss.loss_items.classify.ini.sample_weighted = True
cfg.loss.loss_items.classify.cls = EmbClassifyLoss
cfg.loss.loss_items.classify.cal = EmbClassifyLoss.emb_dt_emb_classifier_cal
cfg.loss.loss_items.classify.weights = 1.

# -* 设定优化器。
cfg.opt.get_pg.ini.lr = 0.02
cfg.opt.get_pg.ini.weight_decay = 0.
cfg.opt.get_pg.cal = EmbLinearProbe.get_pg

cfg.opt.ini.momentum = 0.9
cfg.opt.cls = optim.SGD

cfg.opt.grad_clip.grad_max_norm = None

# -* 设定调率器。
cfg.sched.warm.warm_iters = IL(lambda c: int(6_000 // c.solver.iter_factor), priority=0)  # ~5 epochs
cfg.sched.warm.ini.start_factor = IL(lambda c: 1e-5 / c.opt.get_pg.ini.lr)
cfg.sched.warm.ini.end_factor = 1.
cfg.sched.warm.ini.total_iters = IL(lambda c: c.sched.warm.warm_iters)
cfg.sched.warm.cls = sched.LinearLR

cfg.sched.main.ini.T_max = IL(lambda c: c.solver.max_iter - c.sched.warm.warm_iters)
cfg.sched.main.ini.eta_min = 0.
cfg.sched.main.cls = sched.CosineAnnealingLR

# -* 开启自动混合精度。
cfg.amp.enabled = False
cfg.amp.scaler.ini.enabled = IL(lambda c: c.amp.enabled)
cfg.amp.scaler.ini.init_scale = 2.**16

# -* 设定solver。
cfg.solver.iter_factor = 1
cfg.solver.max_iter = IL(lambda c: int(60_000 // c.solver.iter_factor), priority=0)  # ~50 epochs
cfg.solver.sub_iter_num = 1
cfg.solver.display_step = 100
cfg.solver.loss_average_step = IL(lambda c: c.solver.display_step)
cfg.solver.save_step = IL(lambda c: max(c.solver.max_iter // 1, 1000))
cfg.solver.val_step = IL(lambda c: c.solver.save_step)

# -* 设定验证配置。
cfg.val.cfg = Config(caps='configs/classify_sp/samq,clip/probe.py')

# -** 设定验证配置的路径。
cfg.val.cfg.rslt_dir = '请填入验证结果保存路径'
cfg.val.cfg.emb.dir = IL(lambda c: c.dt.ini.root)

# -** 验证配置不构造分类器。
cfg.val.cfg.classifier.empty_leaf()
