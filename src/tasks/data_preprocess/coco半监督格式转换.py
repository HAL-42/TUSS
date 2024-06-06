#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/9 23:32
@File    : coco半监督格式转换.py
@Software: PyCharm
@Desc    : 
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=Path)
parser.add_argument('--target_dir', type=Path)
args = parser.parse_args()

# -* 搜索src下所有txt文件。
for txt_file in args.src_dir.glob('**/*.txt'):
    img_ids = [Path(line.strip().split()[0]).stem for line in txt_file.open()]

    # -* 生成新的txt文件。
    target_file = args.target_dir / txt_file.relative_to(args.src_dir)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text('\n'.join(img_ids))
