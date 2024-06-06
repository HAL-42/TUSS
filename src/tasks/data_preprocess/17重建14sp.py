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
import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from libs.data.coco2014 import COCO2014

parser = argparse.ArgumentParser()
parser.add_argument('--search_dir', type=Path, required=True)
parser.add_argument('--out_dir', type=Path, required=True)
args = parser.parse_args()

coco_14_t = COCO2014(split='train')
coco_14_v = COCO2014(split='val')

(coco14_t_fine_sp_dir := Path(args.out_dir) / 'train' / 'anns').mkdir(parents=True, exist_ok=True)
(coco14_v_fine_sp_dir := Path(args.out_dir) / 'val' / 'anns').mkdir(parents=True, exist_ok=True)

for img_id in tqdm(coco_14_t.image_ids, desc='Copy train images', unit='img', dynamic_ncols=True):
    fine_sp_pkl = (list(args.search_dir.glob(f'train/anns/{img_id}.pkl')) or
                   list(args.search_dir.glob(f'val/anns/{img_id}.pkl')))
    assert len(fine_sp_pkl) == 1
    fine_sp_pkl = fine_sp_pkl[0]
    shutil.copy2(fine_sp_pkl, coco14_t_fine_sp_dir / f'{img_id}.pkl')

for img_id in tqdm(coco_14_v.image_ids, desc='Copy val images', unit='img', dynamic_ncols=True):
    fine_sp_pkl = (list(args.search_dir.glob(f'train/anns/{img_id}.pkl')) or
                   list(args.search_dir.glob(f'val/anns/{img_id}.pkl')))
    assert len(fine_sp_pkl) == 1
    fine_sp_pkl = fine_sp_pkl[0]
    shutil.copy2(fine_sp_pkl, coco14_v_fine_sp_dir / f'{img_id}.pkl')
