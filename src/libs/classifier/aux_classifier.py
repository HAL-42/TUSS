#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/28 21:04
@File    : aux_classifier.py
@Software: PyCharm
@Desc    : 
"""
import os
import typing as t
from collections import abc
from pathlib import Path

import torch

from libs.classifier.typing import SPEmbClassifier
from libs.data import SAMQEmbDt

__all__ = ['LinearCombineClassifier', 'GatherClassifierResultReader']


class LinearCombineClassifier(object):

    def __init__(self, score_suppress: dict[str, ...]=None, **classifier_cfgs: dict[str, ...]):
        """Combine multiple classifiers with linear weights.

        Args:
            score_suppress: Suppress score.
            classifier_cfgs: M Configs for each classifier.
        """
        self.score_suppress = score_suppress

        self.classifiers: dict[str, SPEmbClassifier] = {name: cfg['cls'](**cfg['ini'])
                                                        for name, cfg in classifier_cfgs.items()}
        self.weights: dict[str, float] = {name: cfg['comb_weight'] for name, cfg in classifier_cfgs.items()}

    def __call__(self, emb: torch.Tensor, img_id: str) -> torch.Tensor:
        """Combine multiple classifiers with linear weights.

        Args:
            emb: [N, D],f4 Embedding.
            img_id: Image id.

        Returns:
            [N, C],f4 Predicted logit.
        """
        logit = None
        suppress_mask = None

        for name, classifier in self.classifiers.items():
            # -* 计算分类器的输出。
            l = classifier(emb, img_id) * self.weights[name]

            # -* 如果分类器输出为score，且需要抑制分类器的0输出（不存在类别）。
            match self.score_suppress:
                case None:
                    pass
                case {'eps': float(eps), 'comb_op': abc.Callable() as comb_op}:
                    m = l < eps
                    suppress_mask = m if suppress_mask is None else comb_op(suppress_mask, m)

            # -* 累加分类器的输出。
            logit = l if logit is None else logit + l

        if suppress_mask is not None:
            logit[suppress_mask] = 0.

        return logit


class GatherClassifierResultReader(object):

    def __init__(self,
                 emb_dir: str | os.PathLike,
                 emb_classify_rslt_dir: str | os.PathLike,
                 dt_name: t.Literal['scores', 'logits']='scores'):
        emb_dt = SAMQEmbDt.from_embwise_h5(emb_dir,
                                           embwise_h5=Path(emb_classify_rslt_dir) / 'embwise.h5',
                                           prob2soft_lb='identical', prob_dt_name=dt_name)

        self.logit = emb_dt.psudo_prob['probs']
        self.img_id_start_end_idx = emb_dt.img_id_start_end_indices

    def __call__(self, emb: torch.Tensor, img_id: str) -> torch.Tensor:
        """Read classifier result from file.

        Args:
            emb: [N,D],f4 Embedding.
            img_id: Image id.

        Returns:
            [N,C],f4 Predicted logit.
        """
        start, end = self.img_id_start_end_idx[img_id]
        return torch.from_numpy(self.logit[start:end]).to(device=emb.device).to(dtype=emb.dtype)
