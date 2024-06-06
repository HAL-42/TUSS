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
parser.add_argument('--src_train', type=Path, required=True)
parser.add_argument('--src_val', type=Path, required=True)
parser.add_argument('--target_train', type=Path, required=True)
parser.add_argument('--target_val', type=Path, required=True)
args = parser.parse_args()

coco_14_t = COCO2014(split='train')
coco_14_v = COCO2014(split='val')

args.target_train.mkdir(parents=True, exist_ok=True)
args.target_val.mkdir(parents=True, exist_ok=True)

for img_id in tqdm(coco_14_t.image_ids, desc='Copy Emb', unit='img', dynamic_ncols=True):
    src_emb = list(args.src_train.glob(f'{img_id}.pkl')) or list(args.src_val.glob(f'{img_id}.pkl'))
    assert len(src_emb) == 1
    src_emb = src_emb[0]
    shutil.copy2(src_emb, args.target_train / f'{img_id}.pkl')

for img_id in tqdm(coco_14_v.image_ids, desc='Copy Emb', unit='img', dynamic_ncols=True):
    src_emb = list(args.src_train.glob(f'{img_id}.pkl')) or list(args.src_val.glob(f'{img_id}.pkl'))
    assert len(src_emb) == 1
    src_emb = src_emb[0]
    shutil.copy2(src_emb, args.target_val / f'{img_id}.pkl')
