#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/8 下午2:53
@File    : cs标签预处理.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import shutil
import sys
from pathlib import Path

from alchemy_cat.contrib.voc.scripts.colorize_voc import colorize_voc
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.data.cityscapes_dt import CityScapes

parser = argparse.ArgumentParser(description='Cityscape标签预处理')
parser.add_argument('--ori_label_dir', type=Path)
parser.add_argument('--search_dir', type=Path)
args = parser.parse_args()

# -* 建立新的标签目录。
(fine_label_dir := args.ori_label_dir.parent / 'fine_label').mkdir(exist_ok=True)

# -* 遍历原标签目录。
for ori_label_file in tqdm(list(args.ori_label_dir.glob('*.png')), desc='处理标签', dynamic_ncols=True):
    # -* 将原始名转为搜索名。譬如 aachen_000000_000019_leftImg8bit -> aachen_000000_000019_gtFine_labelTrainIds
    ori_stem = ori_label_file.stem
    search_name = f'{"_".join(ori_stem.split("_")[:-1])}_gtFine_labelTrainIds.png'

    # -* 在search_dir中递归搜索对应的标签。
    search_label_files = list(args.search_dir.glob(f'**/{search_name}'))
    assert len(search_label_files) == 1, f'找到多个或没有对应的标签文件：{search_label_files}'
    search_label_file = search_label_files[0]

    # -* 将找到的标签复制到新的标签目录，并重命名为ori_stem。
    fine_label_file = fine_label_dir / f'{ori_stem}.png'
    shutil.copy2(search_label_file, fine_label_file)

# -* 上色。
(viz_dir := args.ori_label_dir.parent / 'viz' / 'fine_label').mkdir(parents=True, exist_ok=True)
colorize_voc(str(fine_label_dir), str(viz_dir),
             num_workers=0, is_eval=False,
             l2c=CityScapes.label_map2color_map)
