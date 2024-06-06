#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/10 14:26
@File    : 检查ann是否一致.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.sam import SamAnns

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--sp_dir', type=Path, required=True)
parser.add_argument('-t', '--target_dir', type=Path, required=True)
args = parser.parse_args()

src_sps_count, target_sps_count = 0, 0
src_more_count = 0
for sp_file in tqdm(list(args.sp_dir.glob('**/*.pkl')), desc='检查', dynamic_ncols=True):
    target_sp_file = list(args.target_dir.glob(f'**/{sp_file.name}'))

    if len(target_sp_file) == 0:
        print(f"{sp_file} not in {args.target_dir}")
        continue

    assert len(target_sp_file) == 1, f"More than one {sp_file} in {args.target_dir}"
    target_sp_file = target_sp_file[0]

    src_sps = SamAnns(pickle.loads(sp_file.read_bytes()))
    target_sps = SamAnns(pickle.loads(target_sp_file.read_bytes()))

    src_sps_count += len(src_sps)
    target_sps_count += len(target_sps)

    if len(src_sps) > len(target_sps):
        src_more_count += 1

    if len(src_sps) != len(target_sps):
        print(f"{sp_file} and {target_sp_file} not equal")
        print(f"src有{len(src_sps)}个超像素，target有{len(target_sps)}个超像素")

print(f"src有{src_sps_count}个超像素，target有{target_sps_count}个超像素")
print(f"src有{src_more_count}个样本上超像素比target多")
