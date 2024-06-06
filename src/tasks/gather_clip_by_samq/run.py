#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/29 15:00
@File    : gather_clip_by_mask.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import pickle
import sys
from functools import partial
from pathlib import Path

import open_clip
import torch
from alchemy_cat.alg import divisible_by_n
from alchemy_cat.py_tools import ADict, Config
from alchemy_cat.torch_tools import init_env
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.sam.custom_sam import SamAnns


def main(args: argparse.Namespace, cfg: str | Config):
    # -* 初始化环境。
    _, cfg = init_env(is_cuda=True,
                      is_benchmark=bool(args.benchmark),
                      is_train=False,
                      config_path=cfg,
                      experiments_root="experiment",
                      rand_seed=0,
                      cv2_num_threads=-1,
                      verbosity=True,
                      log_stdout=True,
                      reproducibility=False,
                      is_debug=bool(args.is_debug))

    # -* 配置路径。
    (emb_dir := Path(cfg.rslt_dir) / 'emb').mkdir(parents=True, exist_ok=True)

    # -* 模型。
    clip, _, preprocess = open_clip.create_model_and_transforms(**cfg.clip.ini)

    extractor = cfg.extractor.cls(clip, **cfg.extractor.ini).to('cuda')
    sem_agg = cfg.sem_agg.cls(clip, **cfg.sem_agg.ini).to('cuda')
    print(extractor, sem_agg, sep='\n', end='\n\n')

    # -* 数据集。
    dt = cfg.dt.cls(**cfg.dt.ini)
    print(dt, end="\n\n")

    # -* 训练数据增强器。
    auger_ini = cfg.auger.ini.branch_copy()
    auger_ini.mean = preprocess.transforms[-1].mean
    auger_ini.std = preprocess.transforms[-1].std
    auger_ini.scale_crop_method.aligner = partial(divisible_by_n, n=clip.visual.patch_size[0], direction='larger')
    auger = cfg.auger.cls(dt, **auger_ini)
    print(auger, end="\n\n")

    # -* 数据加载器。
    val_loader = DataLoader(auger,
                            batch_size=1,
                            num_workers=8,
                            pin_memory=False,
                            shuffle=False,
                            drop_last=False,
                            generator=torch.Generator().manual_seed(0),
                            prefetch_factor=2,
                            persistent_workers=False
                            )
    print(val_loader, end="\n\n")

    for inp in tqdm(val_loader, dynamic_ncols=True, desc='推理', unit='批次', miniters=10):
        # -* 获取新一个批次数据。
        img_id, img = inp.img_id[0], inp.img.to(device='cuda', non_blocking=True)
        ori_lb = torch.from_numpy(dt.get_by_img_id(img_id).lb).to(device='cuda', non_blocking=True).to(dtype=torch.long)

        # -* 跳过已有（只对bt=1有效）。
        if (save_pkl := emb_dir / f'{img_id}.pkl').is_file():
            print(f"[跳过] {save_pkl}已存在，跳过。")
            continue

        # -* 读取SAM标注。
        with open(sp_file := (cfg.sam_sps.dir / f'{img_id}.pkl'), 'rb') as pkl_f:
            sps = SamAnns(pickle.load(pkl_f))
        sps.stack_data(masks=True, areas=True, device='cuda', replace=True)

        # -* 获取该超像素基于投票法的分类概率。
        sp_lb_vote_prob = torch.zeros((len(sps), dt.class_num), device='cuda', dtype=torch.float32)
        for i, m in enumerate(sps.data.masks):  # TODO：将lb转为one-hot，这个for可以被优化为批量操作。
            lb_on_mask_ = ori_lb[m]
            lb_on_mask = lb_on_mask_[lb_on_mask_ != dt.ignore_label]
            cls_in_sp, cls_count_in_sp = torch.unique(lb_on_mask, return_counts=True)

            sp_lb_vote_prob[i, cls_in_sp] = cls_count_in_sp.float() / cls_count_in_sp.sum(dtype=torch.float32)

        # -* 为每个超像素提取特征。
        clip_out = extractor(img)
        sp_embs: torch.Tensor = sem_agg(clip_out, [sps.data.masks])[0]  # MxD

        # -* 保存ann信息。
        with open(save_pkl, 'wb') as pkl_f:
            pickle.dump(ADict.from_dict({'sp_embs': sp_embs.cpu().numpy(),
                                         'sp_lb_vote_prob': sp_lb_vote_prob.cpu().numpy(),
                                         'sp_file': sp_file}),  # 便于找回对应的sp。
                        pkl_f)


if __name__ == '__main__':
    # -* 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-b", '--benchmark', default=0, type=int)
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    _args = parser.parse_args()
    main(_args, _args.config)
