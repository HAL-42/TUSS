#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/15 上午1:52
@File    : pca.py
@Software: PyCharm
@Desc    : 
"""
import typing as t
from dataclasses import dataclass

import numpy as np
import torch
from alchemy_cat.alg import samp_idxes_with_2d_mask
from sklearn.decomposition import PCA

from others.WeightedPCA.WeightedPCA import WPCA as WPCA

if t.TYPE_CHECKING:
    from tasks.feature_preprocess.remove_pc.run import RemovePCPacket

__all__ = ['PCARet', 'SamplingPCA', 'WeightedPCA']


@dataclass
class PCARet(object):
    components: np.ndarray = None  # [d,d],f4
    explained_variance: np.ndarray = None  # [d],f4


class SamplingPCA(object):

    def __init__(self, sample_num: int=-1, resample_limit: int=1, rand_seed: int=0):
        self.sample_num = sample_num
        self.resample_limit = resample_limit
        self.rand_seed = rand_seed

    def __repr__(self):
        return f'{self.__class__.__name__}(sample_num={self.sample_num}, resample_limit={self.resample_limit}, ' \
               f'rand_seed={self.rand_seed})'

    def __call__(self, packet: 'RemovePCPacket') -> PCARet:
        X, Y = packet.sp_emb, packet.Y_bar

        if self.sample_num < 0:
            samp_X = X
        else:
            cls = np.unique(Y)  # [K],i8
            cls_mask = cls[:, None] == Y  # [K,N],bool
            samp_idxes = samp_idxes_with_2d_mask(torch.from_numpy(cls_mask),
                                                 samp_nums=self.sample_num, resamp_lim=self.resample_limit,
                                                 g=torch.Generator().manual_seed(self.rand_seed))
            samp_idx: np.ndarray = torch.cat(samp_idxes, dim=0).numpy()

            samp_X = X[samp_idx, :]

        pca = PCA()
        pca.fit(samp_X)

        return PCARet(components=pca.components_, explained_variance=pca.explained_variance_)


class WeightedPCA(object):

    def __call__(self, packet: 'RemovePCPacket') -> PCARet:
        X = packet.sp_emb
        W = packet.weight

        wpca = WPCA(n_components=None, scale=False)
        wpca.fit(X, sample_weight=W)

        return PCARet(components=wpca.components_.T, explained_variance=wpca.exp_var_)
