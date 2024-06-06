#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/24 下午5:00
@File    : sampler.py
@Software: PyCharm
@Desc    : 
"""
import itertools
import typing as t

import numpy as np
import torch
from alchemy_cat.alg import shuffle_cat
from torch.utils.data import Sampler

if t.TYPE_CHECKING:
    from .emb_dt import SAMQEmbDt


class SemiBalanceSampler(Sampler[int]):

    def __init__(self,
                 semi_mask: np.ndarray | torch.Tensor, unlabeled_labeled_num: tuple[int, int]=None,
                 generator: torch.Generator=None):
        super().__init__(None)
        self.semi_mask = torch.as_tensor(semi_mask)
        self.nu, self.nl = unlabeled_labeled_num
        self.generator = generator

        self.unlabeled_indices = torch.nonzero(~self.semi_mask, as_tuple=True)[0]
        self.labeled_indices = torch.nonzero(self.semi_mask, as_tuple=True)[0]

        self.labeled_repeat_factor = max(1.,
                                         (self.nl / self.nu) * (self.unlabeled_len / self.labeled_len))
        self.repeated_labeled_len = round(self.labeled_len * self.labeled_repeat_factor)

    @classmethod
    def from_emb_dt(cls, emb_dt: 'SAMQEmbDt', unlabeled_labeled_num: tuple[int, int]=None,
                    generator: torch.Generator=None):
        assert emb_dt.semi_mask is not None
        return cls(emb_dt.semi_mask, unlabeled_labeled_num, generator)

    @property
    def unlabeled_len(self):
        return self.unlabeled_indices.shape[0]

    @property
    def labeled_len(self):
        return self.labeled_indices.shape[0]

    def __len__(self):
        return self.unlabeled_len + self.repeated_labeled_len

    def __repr__(self):
        return (f'{self.__class__.__name__}(nu={self.nu}, nl={self.nl}, '
                f'unlabeled_len={self.unlabeled_len}, labeled_len={self.labeled_len}, '
                f'repeated_labeled_len={self.repeated_labeled_len})')

    def __iter__(self) -> t.Iterator[int]:
        unlabeled_indices_shuffled = shuffle_cat(self.unlabeled_indices, shuffled_len=None,
                                                 shuffle_all=False, g=self.generator).tolist()
        labeled_indices_repeated_shuffled = shuffle_cat(self.labeled_indices, shuffled_len=self.repeated_labeled_len,
                                                        shuffle_all=False, g=self.generator).tolist()
        assert len(unlabeled_indices_shuffled) + len(labeled_indices_repeated_shuffled) == len(self)

        yield from itertools.chain.from_iterable(
            map(lambda u_l_batch: u_l_batch[0] + u_l_batch[1],
                itertools.zip_longest(self.batched(unlabeled_indices_shuffled, self.nu),
                                      self.batched(labeled_indices_repeated_shuffled, self.nl),
                                      fillvalue=()))
        )

    @staticmethod
    def batched(iterable: t.Iterable[int], n: int=1) -> t.Iterator[tuple[int]]:
        """
        Batch elements of an iterable into chunks of size n,
        e.g. batched('ABCDEFGH', 3) -> 'ABC', 'DEF', 'GH'
        """
        if n <= 0:
            return

        args = [iter(iterable)] * n
        yield from map(lambda batch: tuple(item for item in batch if item is not None),
                       itertools.zip_longest(*args, fillvalue=None))
