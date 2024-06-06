#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/18 16:09
@File    : add_vote_prob_to_sp.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import pickle
import sys
import typing as t
from pathlib import Path

import torch
from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.contrib.voc import VOCAug2
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.data.coco2014 import COCO2014
from libs.seeding.seed_anns.utils import sort_anns, scatter_anns
from libs.data.cityscapes_dt import CityScapes
from libs.data.coco164k_dt import COCOStuff, COCOObj
from libs.sam import SamAnns

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--sp_dir', type=Path, required=True)
parser.add_argument('--dt_type', type=str, default='voc')
parser.add_argument('--ignore_thresh', type=float, default=float('inf'))
parser.add_argument('--replace', type=int, default=1)
parser.add_argument('--priority', type=str, default='level_bigger')
args = parser.parse_args()

match args.dt_type:
    case 'voc':
        dts = [VOCAug2('datasets', split='trainval_aug', rgb_img=False, ret_img_file=True),
               VOCAug2('datasets', split='test', rgb_img=False, ret_img_file=True)]
    case 'cs':
        dts = [CityScapes('datasets/cityscapes', split='train', rgb_img=True, ret_img_file=True),
               CityScapes('datasets/cityscapes', split='val', rgb_img=True, ret_img_file=True)]
    case 'stuff':
        dts = [COCOStuff('datasets/cocostuff164k', split='train', rgb_img=True, ret_img_file=True),
               COCOStuff('datasets/cocostuff164k', split='val', rgb_img=True, ret_img_file=True)]
    case 'obj':
        dts = [COCOObj('datasets/cocostuff164k', split='train', rgb_img=True, ret_img_file=True),
               COCOObj('datasets/cocostuff164k', split='val', rgb_img=True, ret_img_file=True)]
    case 'c14':
        dts = [COCO2014('datasets', split='train', rgb_img=True, ret_img_file=True),
               COCO2014('datasets', split='val', rgb_img=True, ret_img_file=True)]
    case 'c14_from17':
        dts = [COCO2014('datasets', split='train', rgb_img=True, ret_img_file=True,
                        cocostuff_img=True, cocostuff_lb=True),
               COCO2014('datasets', split='val', rgb_img=True, ret_img_file=True,
                        cocostuff_img=True, cocostuff_lb=True)]
    case _:
        raise ValueError(f"Unknown dt_type: {args.dt_type}")

seg_metric = SegmentationMetric(dts[0].class_num, dts[0].class_names)

for sp_file in tqdm(list(args.sp_dir.glob('**/*.pkl')), desc='Adding vote prob to sp', dynamic_ncols=True):
    img_id = sp_file.stem

    # -* 获取原始标签。
    dt = None
    for d in dts:
        if img_id in d.image_ids:
            dt = d

    ori_lb = torch.from_numpy(ori_lb_ := dt.get_by_img_id(img_id).lb).to(device='cuda',
                                                                         non_blocking=True).to(dtype=torch.long)

    # -* 获取超像素。
    with open(sp_file, 'rb') as pkl_f:
        sps = SamAnns(pickle.load(pkl_f))
    sps.stack_data(masks=True, areas=True, device='cuda', replace=bool(args.replace))

    # -* 获取该超像素基于投票法的分类概率。
    sp_lb_vote_prob = torch.zeros((len(sps), dt.class_num), device='cuda', dtype=torch.float32)
    is_ignore = torch.zeros(len(sps), dtype=torch.bool, device='cuda')
    for i, m in enumerate(sps.data.masks):  # TODO：将lb转为one-hot，这个for可以被优化为批量操作。
        lb_on_mask_ = ori_lb[m]

        ignore_m = t.cast(torch.Tensor, lb_on_mask_ == dt.ignore_label)

        # 若ignore面积超过阈值，则所有类别概率都为0。
        if ignore_m.sum(dtype=torch.float32) / ignore_m.numel() > args.ignore_thresh:
            is_ignore[i] = True
            continue

        lb_on_mask = lb_on_mask_[~ignore_m]
        cls_in_sp, cls_count_in_sp = torch.unique(lb_on_mask, return_counts=True)

        sp_lb_vote_prob[i, cls_in_sp] = cls_count_in_sp.float() / cls_count_in_sp.sum(dtype=torch.float32)

    vote_lb = torch.argmax(sp_lb_vote_prob, dim=1)
    vote_lb[is_ignore] = dt.ignore_label

    # -* 保存。
    sps.add_item('sp_lb_vote_prob', sp_lb_vote_prob.cpu().numpy())
    sps.add_item('vote_lb', vote_lb.tolist())
    sps.clear_data()
    with open(sp_file, 'wb') as pkl_f:
        pickle.dump(sps, pkl_f)

    # -* 从sp恢复标签。
    sorted_sps = sort_anns(sps, priority=(args.priority,))
    sorted_sps_seed = torch.as_tensor([sp['vote_lb'] for sp in sorted_sps], dtype=torch.long, device='cuda')
    seed = scatter_anns(sorted_sps_seed, sorted_sps, default_vec=0).cpu().numpy()

    seed[seed == dt.ignore_label] = 0
    seg_metric.update(seed, ori_lb_)

seg_metric.print_statistics(0)
