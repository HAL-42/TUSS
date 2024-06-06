#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/13 21:39
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
from pathlib import Path

from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.contrib.voc import VOCAug2

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--seed_dir', type=Path)
args = parser.parse_args()

dt = VOCAug2('datasets', split=args.split)

preds = [str(pred) for pred in args.seed_dir.iterdir() if pred.stem in dt.image_ids]

metric = eval_preds(class_num=dt.class_num, class_names=dt.class_names, preds_dir=preds,
                    gts_dir=dt.label_dir, importance=0, eval_individually=False)
metric.print_statistics(0)
