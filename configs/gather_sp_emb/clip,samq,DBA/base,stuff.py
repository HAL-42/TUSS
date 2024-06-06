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

from libs.data.coco164k_dt import COCOStuff

cfg = config = Config(caps='configs/gather_sp_emb/clip,samq,DBA/base,obj.py')

# -* 设定数据集。
cfg.dt.cls = COCOStuff

# -* 设定SAM superpixels数据位置。
cfg.sam_sps.dir = DEP(lambda c: Path(f'experiment/sam_auto_seg/c7/{c.dt.ini.split}/anns_stuff'))
