#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/23 下午2:45
@File    : gather_metric.py
@Software: PyCharm
@Desc    : 
"""
import os
import pickle
from pathlib import Path

__all__ = ['read_seg_cls_metric']


def read_seg_cls_metric(eval_dir: str | os.PathLike | Path) -> dict[str, ...]:
    if (Path(eval_dir) / 'semi_seg').is_dir():  # 如果是半监督任务，优先读取无标签数据的评估结果。
        seg_metric = pickle.loads((Path(eval_dir) / 'semi_seg' / 'statistics.pkl').read_bytes())
        cls_metric = pickle.loads((Path(eval_dir) / 'semi_cls' / 'statistics.pkl').read_bytes())
    else:
        seg_metric = pickle.loads((Path(eval_dir) / 'seg' / 'statistics.pkl').read_bytes())
        cls_metric = pickle.loads((Path(eval_dir) / 'cls' / 'statistics.pkl').read_bytes())

    metrics = {'mIoU': seg_metric['mIoU'],
               'seg_precision': seg_metric['macro_avg_precision'],
               'seg_recall': seg_metric['macro_avg_recall'],
               'seg_accuracy': seg_metric['accuracy'],
               'seg_cls_IoU': seg_metric['cls_IoU'],
               'cls_F1': cls_metric['F1_score'],
               'cls_precision': cls_metric['macro_avg_precision'],
               'cls_recall': cls_metric['macro_avg_recall'],
               'cls_accuracy': cls_metric['accuracy']}

    return metrics
