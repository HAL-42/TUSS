#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 14:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from pathlib import Path

from alchemy_cat.py_tools import Config, parse_config, auto_rslt_dir, IL

gather_cfg = parse_config(str(Path(auto_rslt_dir(__file__, '', trunc_cwd=True)) / '..' / '..' / 'cfg'),
                          create_rslt_dir=False)

cfg = config = Config('configs/classify_sp/samq,clip/base.py')

cfg.clip = gather_cfg.clip.branch_copy()

cfg.emb.dir = IL(lambda c: Path(c.rslt_dir) / '..' / '..' / 'emb')
