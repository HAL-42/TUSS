#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/13 3:23
@File    : 17重建14图.py
@Software: PyCharm
@Desc    : 
"""
import shutil
from pathlib import Path

from tqdm import tqdm

from libs.data.coco164k_dt import COCOStuff
from libs.data.coco2014 import COCO2014

coco_14_t = COCO2014(split='train')
coco_14_v = COCO2014(split='val')

coco_stuff_t = COCOStuff(split='train')
coco_stuff_v = COCOStuff(split='val')

(coco14_t_fine_dir := Path(coco_14_t.image_dir.replace('/images/', '/images_fine/'))).mkdir(parents=True,
                                                                                            exist_ok=True)
(coco14_v_fine_dir := Path(coco_14_v.image_dir.replace('/images/', '/images_fine/'))).mkdir(parents=True,
                                                                                            exist_ok=True)

for img_id in tqdm(coco_14_t.image_ids, desc='Copy train images', unit='img', dynamic_ncols=True):
    dt: COCOStuff = coco_stuff_t if img_id in coco_stuff_t.image_ids else coco_stuff_v
    shutil.copy2(dt.image_dir / f'{img_id}.jpg', coco14_t_fine_dir / f'{img_id}.jpg')

for img_id in tqdm(coco_14_v.image_ids, desc='Copy val images', unit='img', dynamic_ncols=True):
    dt: COCOStuff = coco_stuff_t if img_id in coco_stuff_t.image_ids else coco_stuff_v
    shutil.copy2(dt.image_dir / f'{img_id}.jpg', coco14_v_fine_dir / f'{img_id}.jpg')
