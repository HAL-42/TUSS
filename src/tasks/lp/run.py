#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 13:26
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import gc
import multiprocessing as mp
import os
import pickle
import sys
import typing as t
from collections import abc
from dataclasses import dataclass
from pathlib import Path
from pprint import pp

import numpy as np
import scipy
import torch
from alchemy_cat.py_tools import Config, ADict
from alchemy_cat.torch_tools import init_env
from loguru import logger

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.classifier.typing import SPEmbClassifier
from libs.label_propagation import LPClassifier, LabelPropagator
from libs.seeding.confidence import area_conf_weight
from libs.data import SAMQEmbDt
from utils.cls_lb import cls_lb2fg_cls
from libs.sam.custom_sam import SamAnns
from tasks.classify_samq_emb import run as classify_samq_emb_run


@dataclass
class LPPacket(object):
    # -* 输入量。
    emb_dir: Path = None
    sp_emb: np.ndarray = None  # [N,d],f4
    area: np.ndarray = None  # [N,],i4
    emb_dt: SAMQEmbDt = None
    score: np.ndarray = None  # [N,K],f4
    cls_lb: np.ndarray = None  # [N,K],?
    semi_ids: list[str] = None
    semi_mask: np.ndarray = None

    # -* 中间量。
    X: np.ndarray = None  # [N,d],f4
    Y: np.ndarray = None  # [N,K],f4
    degree: np.ndarray = None  # [N],f4
    A: scipy.sparse.csr_matrix | None = None  # [N,N],f4，LP公式的系数矩阵。

    # -* 输出量。
    _lp_score: np.ndarray | None = None  # [N,K],f4
    lp_score_for_seed: np.ndarray | None = None  # [N,K],f4

    def save_A_degree(self, A_degree_dir: str | os.PathLike):
        Path(A_degree_dir).mkdir(parents=True, exist_ok=True)
        scipy.sparse.save_npz(Path(A_degree_dir) / 'A.npz', self.A)
        np.save(Path(A_degree_dir) / 'degree.npy', self.degree)

    @classmethod
    def from_gather_and_classify_result(cls,
                                        emb_dir: str | os.PathLike,
                                        emb_classify_rslt_dir: str | os.PathLike | None,
                                        semi_ids: str | os.PathLike | list[str]=None) -> 'LPPacket':
        # NOTE 如果做特征预处理，生成假的emb_dir和emb_classify_rslt_dir即可。
        # NOTE 如果特征预处理后，sp_emb增加，则扩展在后面。修改本类，使得lp_score_per_sample依旧是正确的。
        emb_dt = SAMQEmbDt.from_embwise_h5(emb_dir,
                                           embwise_h5=Path(emb_classify_rslt_dir) / 'embwise.h5'
                                           if emb_classify_rslt_dir else None,
                                           prob2soft_lb='identical',
                                           semi_ids=semi_ids)
        sp_emb = np.array(emb_dt.emb_data['sp_embs'], copy=False)
        area = np.array(emb_dt.emb_data['areas'], copy=False)

        return cls(emb_dir=Path(emb_dir),
                   sp_emb=sp_emb, area=area,
                   emb_dt=emb_dt,
                   score=emb_dt.probs, cls_lb=emb_dt.cls_lb,
                   semi_ids=emb_dt.semi_ids, semi_mask=emb_dt.semi_mask)

    @dataclass
    class SeedingPacket(object):
        sps: SamAnns
        score: torch.Tensor
        fg_cls: torch.Tensor

    @staticmethod
    def _load_sp(sp_file: str) -> SamAnns:
        return pickle.loads(Path(sp_file).read_bytes())

    @property
    def lp_score(self) -> np.ndarray:
        return self._lp_score

    @lp_score.setter
    def lp_score(self, value: np.ndarray | None):
        if value is None:
            self._lp_score = None
            self.lp_score_for_seed = None
            return

        self._lp_score = value

        if self.semi_ids is not None:
            self.lp_score_for_seed = value.copy()
            self.lp_score_for_seed[self.semi_mask, :] = self.emb_dt.emb_data['probs'][self.semi_mask, :]
        else:
            self.lp_score_for_seed = value

    @property
    def lp_score_for_seed_per_sample(self) -> t.Generator[torch.Tensor, None, None]:
        for start, end in self.emb_dt.img_id_start_end_indices.values():
            yield torch.as_tensor(self.lp_score_for_seed[start:end],
                                  dtype=torch.float32).to(device='cuda', non_blocking=True)

    @property
    def iter_sample_for_seeding(self) -> t.Generator[SeedingPacket, None, None]:
        with mp.Pool(4) as pool:
            all_sps: t.Iterable[SamAnns] = pool.imap(self._load_sp, self.emb_dt.emb_meta['sp_files'], chunksize=10)
            for lp_score, sps, start_idx in zip(self.lp_score_for_seed_per_sample, all_sps,
                                                self.emb_dt.emb_meta['img_id_start_indexes'], strict=True):
                fg_cls = cls_lb2fg_cls(torch.as_tensor(self.cls_lb[start_idx])).to(
                    device='cuda', non_blocking=True).to(dtype=torch.long)
                yield self.SeedingPacket(sps, lp_score, fg_cls)

    @property
    def dummy_classifier(self) -> SPEmbClassifier:
        img_id_start_end_indices = self.emb_dt.img_id_start_end_indices
        lp_score_for_seed = self.lp_score_for_seed

        def classifier(_: torch.Tensor, img_id: str) -> torch.Tensor:
            start, end = img_id_start_end_indices[img_id]
            return torch.as_tensor(lp_score_for_seed[start:end]).to(device='cuda', non_blocking=True)

        return classifier


def main(_: argparse.Namespace, cfg: str | Config) -> ADict | None:
    # -* 初始化标签传播器。
    label_propagator: LabelPropagator = cfg.lp.cls(**cfg.lp.ini)

    # -* 构造标签传播数据包。
    logger.info("构造标签传播数据包...")
    lp_packet = LPPacket.from_gather_and_classify_result(**cfg.lp_packet.ini)

    # -* 获取X。
    logger.info("初始化特征X...")
    lp_packet.X = LabelPropagator.X_preprocess(lp_packet.sp_emb, cfg.X_method)

    # -* 获取Y。
    logger.info("初始化软标签Y...")
    match cfg.Y_method:
        case None | {'method': 'identical'}:
            lp_packet.Y = lp_packet.score
        case {'method': 'area_conf_weighted', 'alpha': float(alpha),
              'area': {'norm': abc.Callable() as area_norm, 'area_gamma': float(area_gamma)},
              'conf': {'cal': abc.Callable() as conf_cal, 'norm': abc.Callable() as conf_norm,
                       'conf_gamma': float(conf_gamma)}
              }:
            weight = area_conf_weight(lp_packet.area, lp_packet.score,
                                      alpha,
                                      area_norm, area_gamma,
                                      conf_cal, conf_norm, conf_gamma)

            # -* 计算软标签。
            lp_packet.Y = lp_packet.score * weight[:, None]  # [N,K],f4
        case {'method': 'cls_balance'}:
            lp_packet.Y = LabelPropagator.cls_balance(lp_packet.score)
        case {'method': 'bg_fg_balance', 'balance_ratio': str(balanced_ratio)}:
            lp_packet.Y = LabelPropagator.bg_fg_balance(lp_packet.score, balanced_ratio=balanced_ratio)
        case {'method': 'alpha_bg', 'alpha_bg': float(alpha_bg)}:
            lp_packet.Y = LabelPropagator.alpha_bg(lp_packet.score, alpha_bg)
        case {'method': 'semi_balance', 'balance_ratio': float(balanced_ratio), **others}:
            bg_fg_balance = others.get('bg_fg_balance', None)
            lp_packet.Y = LabelPropagator.semi_balance(lp_packet.score, lp_packet.semi_mask,
                                                       balanced_ratio=balanced_ratio)
            if bg_fg_balance:
                lp_packet.Y = LabelPropagator.bg_fg_balance(lp_packet.Y, balanced_ratio=bg_fg_balance)
        case _:
            raise ValueError(f"Unknown Y_method: {cfg.Y_method}")

    # -* 标签传播。
    lp_packet = cfg.lp.cal(label_propagator, lp_packet)
    # -** 若只是为了预算A、degree，到这里可以保存、返回。
    if label_propagator.A_degree_only:
        lp_packet.save_A_degree(Path(cfg.rslt_dir) / 'A_degree')
        return None

    # -* 生成种子点。
    logger.info("生成种子点...")
    cls_cfg = cfg.cls_cfg.unfreeze().parse(experiments_root='').compute_item_lazy().freeze()
    if lp_packet.semi_ids is not None:
        assert cls_cfg.cls_eval_dt.ini.val
        assert cls_cfg.semi_ids == cfg.lp_packet.ini.semi_ids
    pp(cls_cfg)

    torch.cuda.empty_cache()
    del lp_packet.X
    del lp_packet.sp_emb
    del lp_packet.emb_dt.emb_data
    gc.collect()
    cls_rslt = classify_samq_emb_run.main(argparse.Namespace(), cls_cfg, classifier=lp_packet.dummy_classifier)

    # -* 保存lp推理初始化数据包。
    (LPClassifier.LPClassifierIniPacket.collect_from_lp_run(lp_packet=lp_packet,
                                                            X_method=cfg.X_method, k=label_propagator.k,
                                                            seed_Y_star_rslt_dir=Path(cls_cfg.rslt_dir) /
                                                                                 'emb_classify_rslt').
     dump(Path(cfg.rslt_dir) / 'lp_classifier_ini'))

    return cls_rslt


if __name__ == '__main__':
    # -* 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    _args = parser.parse_args()

    # -* 初始化环境。
    _, _cfg = init_env(is_cuda=True,
                       is_benchmark=False,
                       is_train=False,
                       config_path=_args.config,
                       rand_seed=0,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       loguru_ini=True,
                       reproducibility=False,
                       is_debug=bool(_args.is_debug))

    main(_args, _cfg)
