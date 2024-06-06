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

import torch
from alchemy_cat.alg import divisible_by_n
from alchemy_cat.py_tools import ADict, Config
from alchemy_cat.torch_tools import init_env
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.clip.utils import OpenCLIPCreated
from libs.sam.custom_sam import SamAnns


def main(_: argparse.Namespace, cfg: str | Config, open_clip_created: OpenCLIPCreated=None) -> ADict:
    # -* 配置路径。
    (emb_dir := Path(cfg.rslt_dir) / 'emb').mkdir(parents=True, exist_ok=True)

    # -* 模型。
    if open_clip_created is None:
        open_clip_created = OpenCLIPCreated.create(**cfg.clip.ini)
    clip, preprocess = open_clip_created.model, open_clip_created.preprocess_val

    extractor = cfg.extractor.cls(clip, **cfg.extractor.ini).to('cuda')
    print(extractor, end='\n\n')

    # -* 数据集。
    dt = cfg.dt.cls(**cfg.dt.ini)
    print(dt, end="\n\n")

    # -* 训练数据增强器。
    auger_ini = cfg.auger.ini.branch_copy()
    auger_ini.mean = preprocess.transforms[-1].mean
    auger_ini.std = preprocess.transforms[-1].std
    # NOTE 若没有auger初始化不需要aligner，则理论上下面项无效。
    auger_ini.scale_crop_method.aligner = partial(divisible_by_n, n=clip.visual.patch_size[0], direction='larger')
    auger = cfg.auger.cls(dt, **auger_ini)
    print(auger, end="\n\n")

    # -* 数据加载器。
    loader = DataLoader(auger,
                        batch_size=1,
                        num_workers=8,
                        pin_memory=False,
                        shuffle=False,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(0),
                        prefetch_factor=2,
                        persistent_workers=False)
    print(loader, end="\n\n")

    for inp in tqdm(loader, dynamic_ncols=True, desc='推理', unit='批次', miniters=10):
        # -* 获取新一个批次数据。
        img_id, img = inp.img_id[0], inp.img.to(device='cuda', non_blocking=True)

        # -* 跳过已有（只对bt=1有效）。
        if (save_pkl := emb_dir / f'{img_id}.pkl').is_file():
            print(f"[跳过] {save_pkl}已存在，跳过。")
            continue

        # -* 读取SAM标注。
        with open(sp_file := (cfg.sam_sps.dir / f'{img_id}.pkl'), 'rb') as pkl_f:
            sps = SamAnns(pickle.load(pkl_f))

        # -* 为每个超像素提取特征。
        sp_embs: torch.Tensor = extractor(img, [sps.cuda_masks], [sps])[0]  # MxD

        # -* 保存ann信息。
        with open(save_pkl, 'wb') as pkl_f:
            pickle.dump(ADict.from_dict({'sp_embs': sp_embs.to(dtype=torch.float32).cpu().numpy(),
                                         'sp_file': sp_file}),  # 便于找回对应的sp。
                        pkl_f)

    return ADict({'open_clip_created': open_clip_created})


if __name__ == '__main__':
    # -* 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-b", '--benchmark', default=0, type=int)
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    _args = parser.parse_args()

    # -* 初始化环境。
    _, _cfg = init_env(is_cuda=True,
                       is_benchmark=bool(_args.benchmark),
                       is_train=False,
                       config_path=_args.config,
                       experiments_root="experiment",
                       rand_seed=0,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=bool(_args.is_debug))

    main(_args, _cfg)
