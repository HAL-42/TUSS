#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/28 15:03
@File    : lp_classifier.py
@Software: PyCharm
@Desc    : 
"""
import os
import pickle
import typing as t
from collections import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn.preprocessing as prep
import torch
from loguru import logger

from libs.classifier.typing import SPEmbClassifier
from libs.data import SAMQEmbDt
from libs.seeding.confidence import area_conf_weight
from .label_propagator import LabelPropagator

if t.TYPE_CHECKING:
    from tasks.lp.run import LPPacket

__all__ = ['LPClassifier']


class LPClassifier(object):

    def __init__(self,
                 sp_emb: np.ndarray, area: np.ndarray, X_method: dict[str, ...],
                 degree: np.ndarray, k: int | tuple[int, int],
                 raw_Y_star: np.ndarray, seed_Y_star: np.ndarray, cls_lb: np.ndarray,
                 Y_star_method: dict[str, ...], k_bar: int | tuple[int, int], norm_a_method: dict[str, ...],
                 suppress_no_fg_gather: bool=False, semi_mask: np.ndarray=None):
        """基于Label Propagation的分类器。

        Args:
            sp_emb: [N,d],f4 原始特征。
            area: [N,],i4 区域面积。
            X_method: 特征预处理方法。
            degree: [N],f4 邻接矩阵的度。
            k: 原始LabelPropagator的k。
            raw_Y_star: [N,C],f4 原始lp结果。
            seed_Y_star: [N,C],f4 seeding后的lp结果。
            cls_lb: [N,C],bool 类别标签。
            Y_star_method: lp结果的处理方法。
            k: int 最近邻数。
            k_bar: int 最近邻数。
            norm_a_method: 相似度的归一化方法。
            suppress_no_fg_gather: 考虑k邻近的cls_lb，对k邻近cls_lb中都不存在的类别，y*在该类别上严格置0。
            semi_mask: [N,],bool 半监督掩码。
        """
        if X_method != {'method': 'L2'}:
            raise ValueError(f"目前只支持X_method为L2，但传入的X_method为{X_method}。")

        if not isinstance(k_bar, int):
            assert semi_mask is not None

        # -* 记录参数。
        self.X_method = X_method
        self.degree = degree
        self.k = k
        self.cls_lb = cls_lb
        self.semi_mask = semi_mask

        self.k_bar = k_bar
        self.norm_a_method = norm_a_method
        self.suppress_no_fg_gather = suppress_no_fg_gather

        # -* 计算X和Y_star。
        self.X = self.emb2X(sp_emb, self.X_method)
        self.Y_star = self.get_Y_star(raw_Y_star, seed_Y_star, cls_lb, area, semi_mask, Y_star_method)

        # -* 获取形状信息。
        logger.info(f'{self} start: X.shape={self.X.shape}, Y_star.shape={self.Y_star.shape}')

        # -* 构建faiss索引器。
        # 若k为int，不分块则不传入semi_mask。
        self.index = LabelPropagator.build_faiss_index(self.X,
                                                       semi_mask=None if isinstance(self.k_bar, int) else semi_mask)

    @staticmethod
    def emb2X(sp_emb: np.ndarray, X_method: dict[str, ...]) -> np.ndarray:
        match X_method:
            case {'method': 'L2'}:
                X = prep.normalize(sp_emb, norm='l2')  # [N,d],f4
            case _:
                raise ValueError(f"Unknown X_method: {X_method}")

        return X

    @staticmethod
    def get_Y_star(raw_Y_star: np.ndarray, seed_Y_star: np.ndarray, cls_lb: np.ndarray, area: np.ndarray,
                   semi_mask: np.ndarray | None,
                   Y_star_method: dict[str, ...]) -> np.ndarray:
        match Y_star_method:
            case {'method': 'raw_Y_star'}:
                Y_star = raw_Y_star
            case {'method': 'raw_Y_star_suppress_non_fg'}:
                Y_star = raw_Y_star * cls_lb.astype(np.float32)
            case {'method': 'seed_Y_star'}:
                Y_star = seed_Y_star
            case {'method': 'seed_Y_star_area_conf_weighted', 'alpha': float(alpha),
                  'area': {'norm': abc.Callable() as area_norm, 'area_gamma': float(area_gamma)},
                  'conf': {'cal': abc.Callable() as conf_cal, 'norm': abc.Callable() as conf_norm,
                           'conf_gamma': float(conf_gamma)}}:
                weight = area_conf_weight(area, seed_Y_star,
                                          alpha,
                                          area_norm, area_gamma,
                                          conf_cal, conf_norm, conf_gamma)
                Y_star = seed_Y_star * weight[:, None]
            case {'method': 'seed_Y_star_raw_Y_star_L1_weighted',
                  'raw_Y_star_suppress_non_fg': bool(raw_Y_star_suppress_non_fg)}:
                weight = np.linalg.norm(raw_Y_star * cls_lb.astype(np.float32) if raw_Y_star_suppress_non_fg
                                        else raw_Y_star,
                                        ord=1, axis=1)
                Y_star = seed_Y_star * weight[:, None]
            case {'method': 'semi_balance_seed_Y_star', 'balance_ratio': float(balanced_ratio)}:
                assert semi_mask is not None
                Y_star = LabelPropagator.semi_balance(seed_Y_star, semi_mask,
                                                      balanced_ratio=balanced_ratio)
            case _:
                raise ValueError(f"Unknown Y_star_method: {Y_star_method}")

        return Y_star

    def __call__(self, emb: np.ndarray):
        x = self.emb2X(emb, self.X_method)  # [n,d],f4

        # -* 获取最近邻。
        a, I = self.index.search(x, self.k_bar)
        a: np.ndarray  # [n,K_bar],f4
        I: np.ndarray  # [n,K_bar],i4

        # -* 归一化a。
        match self.norm_a_method:
            case 'L1':
                a_bar = prep.normalize(a, norm='l1', axis=1)  # [n,K_bar],f4
            case 'rigorous':
                d = self.degree[I]  # [n,K],f4
                l1_a = np.linalg.norm(a, ord=1, axis=1, keepdims=True)  # [n,1],f4
                k_bar_correction = ((self.k_bar if isinstance(self.k_bar, int) else sum(self.k_bar)) /
                                    (self.k if isinstance(self.k, int) else (sum(self.k) - 1)))
                denominator = (0.5 * k_bar_correction * d * l1_a) ** (-0.5)  # [n,K_bar],f4
                a_bar = a * denominator
            case _:
                raise ValueError(f"Unknown norm_a_method: {self.norm_a_method}")

        # -* 算出bias_y_star。
        y_star = self.Y_star[I, :]  # [n,K_bar,C],f4
        bias_y_star = np.einsum('nk,nkc->nc', a_bar, y_star)  # [n,C],f4

        # -* 对于k邻近的cls_lb，对k邻近cls_lb中都不存在的类别，y*在该类别上严格置0。
        if self.suppress_no_fg_gather:
            has_cls_lb = self.cls_lb[I, :].any(axis=1)  # [n,C],bool
            bias_y_star[~has_cls_lb] = 0.

        return bias_y_star

    def __repr__(self):
        return (f'LPClassifier(k={self.k}, k_bar={self.k_bar}, '
                f'X_method={self.X_method}, norm_a_method={self.norm_a_method}): \n'
                f'X.shape={self.X.shape}, Y_star.shape={self.Y_star.shape}')

    def classify_samq_emb_cal(self, emb: torch.Tensor, _: str) -> torch.Tensor:
        """emb分类器的输入数据处理。"""
        return torch.from_numpy(self(emb.cpu().numpy())).to(device=emb.device).to(dtype=emb.dtype)

    @classmethod
    def samq_emb_classifier(cls,
                            sp_emb: np.ndarray, area: np.ndarray, X_method: dict[str, ...],
                            degree: np.ndarray, k: int | tuple[int, int],
                            raw_Y_star: np.ndarray, seed_Y_star: np.ndarray, cls_lb: np.ndarray,
                            Y_star_method: dict[str, ...], k_bar: int | tuple[int, int], norm_a_method: dict[str, ...],
                            suppress_no_fg_gather: bool=False, semi_mask: np.ndarray=None) -> SPEmbClassifier:
        """得到classify_samq_emb中的分类器。"""
        classifier = cls(sp_emb=sp_emb, area=area, X_method=X_method,
                         degree=degree, k=k, raw_Y_star=raw_Y_star,
                         seed_Y_star=seed_Y_star, cls_lb=cls_lb,
                         Y_star_method=Y_star_method, k_bar=k_bar, norm_a_method=norm_a_method,
                         suppress_no_fg_gather=suppress_no_fg_gather, semi_mask=semi_mask)

        return classifier.classify_samq_emb_cal

    @dataclass
    class LPClassifierIniPacket(object):
        emb_dir: Path
        X_method: dict[str, ...]
        k: int | tuple[int, int]
        degree: np.ndarray
        raw_Y_star: np.ndarray
        seed_Y_star_rslt_dir: Path
        semi_ids: list[str] | None

        @classmethod
        def collect_from_lp_run(cls,
                                lp_packet: 'LPPacket',
                                X_method: dict[str, ...], k: int | tuple[int, int],
                                seed_Y_star_rslt_dir: str | os.PathLike):
            return cls(emb_dir=lp_packet.emb_dir, X_method=X_method, k=k, degree=lp_packet.degree,
                       raw_Y_star=lp_packet.lp_score, seed_Y_star_rslt_dir=Path(seed_Y_star_rslt_dir),
                       semi_ids=lp_packet.semi_ids)

        def dump(self, save_dir: str | os.PathLike):
            (save_dir := Path(save_dir)).mkdir(parents=True, exist_ok=True)
            (save_dir / 'ini.pkl').write_bytes(pickle.dumps(self))

    @classmethod
    def samq_emb_classifier_from_lp_run(cls, packet: LPClassifierIniPacket | str | os.PathLike,
                                        Y_star_method: dict[str, ...], k_bar: int | tuple[int, int],
                                        norm_a_method: dict[str, ...], suppress_no_fg_gather: bool = False,
                                        k: int=None
                                        ) -> SPEmbClassifier:
        if isinstance(packet, (str, os.PathLike)):
            packet = pickle.loads(Path(packet).read_bytes())

        emb_dt = SAMQEmbDt.from_embwise_h5(packet.emb_dir,
                                           embwise_h5=packet.seed_Y_star_rslt_dir / 'embwise.h5',
                                           prob2soft_lb='identical',
                                           semi_ids=packet.semi_ids if hasattr(packet, 'semi_ids') else None)
        sp_emb = np.asarray(emb_dt.emb_data['sp_embs'])
        area = np.asarray(emb_dt.emb_data['areas'])

        return cls.samq_emb_classifier(sp_emb=sp_emb, area=area, X_method=packet.X_method,
                                       degree=packet.degree, k=packet.k if k is None else k,
                                       raw_Y_star=packet.raw_Y_star,
                                       seed_Y_star=emb_dt.probs, cls_lb=emb_dt.cls_lb,
                                       Y_star_method=Y_star_method, k_bar=k_bar, norm_a_method=norm_a_method,
                                       suppress_no_fg_gather=suppress_no_fg_gather, semi_mask=emb_dt.semi_mask)

    @classmethod
    def samq_emb_classifier_from_emb_fs(cls, emb_dir: str | os.PathLike, k: int,
                                        Y_star_method: dict[str, ...], norm_a_method: dict[str, ...],
                                        k_bar: int=None,
                                        X_method: dict[str, ...]=None, suppress_no_fg_gather: bool=False
                                        ) -> SPEmbClassifier:
        # -* 参数处理。
        if X_method is None:
            X_method = {'method': 'L2'}

        if k_bar is None:
            k_bar = k

        # -* 从emb中获取特征和真值。
        emb_dt = SAMQEmbDt(emb_dir)
        sp_emb = np.asarray(emb_dt.emb_data['sp_embs'])
        area = np.asarray(emb_dt.emb_data['areas'])
        seed_Y_star = np.asarray(emb_dt.emb_data['probs'])

        cls_lb = seed_Y_star > 1e-6

        raw_Y_star = np.broadcast_to(np.full((1, 1), np.nan, dtype=seed_Y_star.dtype), seed_Y_star.shape)

        # -* 获取degree。
        X = LabelPropagator.X_preprocess(sp_emb, X_method)
        _, degree, _ = LabelPropagator.get_Wn(X, k=k, gamma=1., alpha=None)

        return cls.samq_emb_classifier(sp_emb=sp_emb, area=area, X_method=X_method,
                                       degree=degree, k=k, raw_Y_star=raw_Y_star,
                                       seed_Y_star=seed_Y_star, cls_lb=cls_lb,
                                       Y_star_method=Y_star_method, k_bar=k_bar, norm_a_method=norm_a_method,
                                       suppress_no_fg_gather=suppress_no_fg_gather)
