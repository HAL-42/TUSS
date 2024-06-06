#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/10 3:05
@File    : sp_level_sub.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import pickle
import sys
from pathlib import Path

from segment_anything.utils.amg import mask_to_rle_pytorch, area_from_rle
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.sam import SamAnns

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--sp_dir', type=Path, required=True)
args = parser.parse_args()

for sp_file in tqdm(list(args.sp_dir.glob('**/*_*.pkl')), desc='Adding vote prob to sp', dynamic_ncols=True):

    # -* 获取超像素。
    with open(sp_file, 'rb') as pkl_f:
        sps = SamAnns(pickle.load(pkl_f))

    # -* 取出0、1级的超像素mask，合并。
    l0_masks_any = sps.cuda_masks[sps.cuda_levels == 0].any(dim=0)
    l01_masks_any = sps.cuda_masks[sps.cuda_levels == 1].any(dim=0) | l0_masks_any

    # -* 高级别mask挖去低级别mask。
    for sp, m in zip(sps, sps.cuda_masks, strict=True):
        if sp['level'] == 0:
            continue

        if sp['level'] == 1:
            new_m = m & ~l0_masks_any
        else:
            new_m = m & ~l01_masks_any

        sp["segmentation_mode"] = 'uncompressed_rle'
        sp["segmentation"] = mask_to_rle_pytorch(new_m[None, ...])[0]
        sp["area"] = area_from_rle(sp["segmentation"])

    # -* 保存。
    sps.clear_data()
    with open(sp_file, 'wb') as pkl_f:
        pickle.dump(sps, pkl_f)
