#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/3/17 17:09
@File    : label_propagation.py
@Software: PyCharm
@Desc    : 
"""
import os
import typing as t
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import scipy
import sklearn.preprocessing as prep
import torch
from loguru import logger
from tqdm import tqdm
import cupy as cp
import cupyx.scipy as c_scipy
import cupyx.scipy.sparse.linalg as c_linalg

if t.TYPE_CHECKING:
    from tasks.lp.run import LPPacket

__all__ = ['LabelPropagator']


class LabelPropagator(object):

    def __init__(self, k: int | tuple[int, int] | None=50, max_iter: int=200,
                 alpha: float | None=0.99, gamma: float | None=3., cg_guess: bool=False,
                 A_degree_only: bool=False, cupy: bool=False):
        if k is not None:
            assert all(k_ > 0 for k_ in ((k,) if isinstance(k, int) else k))

        self.k = k
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.cg_guess = cg_guess
        self.A_degree_only = A_degree_only
        self.cupy = cupy

        self.A: scipy.sparse.csr_matrix | None = None
        self.degree: np.ndarray | None = None

    @classmethod
    def from_A_degree(cls, A_degree_dir: os.PathLike | str,
                      max_iter: int=200, cg_guess: bool=False,
                      A_degree_only: bool=False, cupy: bool=False) -> 'LabelPropagator':
        logger.info(f'从 {A_degree_dir} 加载A、degree...')
        A = scipy.sparse.load_npz(Path(A_degree_dir) / 'A.npz')
        degree = np.load(Path(A_degree_dir) / 'degree.npy')

        self = cls(k=None, max_iter=max_iter, alpha=None, gamma=None, cg_guess=cg_guess,
                   A_degree_only=A_degree_only, cupy=cupy)
        self.A = A
        self.degree = degree
        return self

    def __repr__(self):
        return f'LabelPropagation(k={self.k}, max_iter={self.max_iter}, alpha={self.alpha}, gamma={self.gamma})'

    @dataclass
    class LPIndices(object):
        indices: tuple[faiss.GpuIndexFlatIP, ...]
        block_Is: tuple[np.ndarray, ...] = None

        def search(self, X: np.ndarray, ks: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            if isinstance(ks, int):
                ks = (ks,)

            # -* 获取每个index下的最近邻及其相似度。
            a_, I_ = zip(*(index.search(X, k) for index, k in zip(self.indices, ks, strict=True)))

            if len(self.indices) == 1:
                # -* 若只有一个index，直接返回。
                return a_[0], I_[0]
            else:
                # -* 若有多个index，先合并。
                a = np.concatenate(a_, axis=1)
                I = np.concatenate([block_I[i] for i, block_I in zip(I_, self.block_Is, strict=True)], axis=1)

                # -* 合并后按照a的大小，从大到小排序。
                sort_idx = np.argsort(a, axis=1)[:, ::-1]
                a = np.take_along_axis(a, sort_idx, axis=1)
                I = np.take_along_axis(I, sort_idx, axis=1)

                return a, I

    @staticmethod
    def _build_single_faiss_index(X: np.ndarray) -> faiss.GpuIndexFlatIP:
        d = X.shape[1]  # 特征维度。

        # -* 构建faiss索引器。
        logger.info(f'构造有 {X.shape[0]} 个key的faiss索引器: d={d}...')
        res = faiss.StandardGpuResources()  # 获取GPU资源。
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = torch.cuda.current_device()  # 使用默认GPU。
        index = faiss.GpuIndexFlatIP(res, d, flat_config)  # 构建索引。

        # -* 建立向量数据库。
        logger.info(f'建立向量数据库...')
        index.add(X)

        return index

    @staticmethod
    def build_faiss_index(X: np.ndarray, semi_mask: np.ndarray=None) -> LPIndices:
        if semi_mask is None:
            index = LabelPropagator._build_single_faiss_index(X)
            indices = LabelPropagator.LPIndices(indices=(index,))
        else:
            labeled_X, unlabeled_X = X[semi_mask], X[~semi_mask]
            labeled_index = LabelPropagator._build_single_faiss_index(labeled_X)
            unlabeled_index = LabelPropagator._build_single_faiss_index(unlabeled_X)

            labeled_indices = np.nonzero(semi_mask)[0]
            unlabeled_indices = np.nonzero(~semi_mask)[0]

            indices = LabelPropagator.LPIndices(indices=(labeled_index, unlabeled_index),
                                                block_Is=(labeled_indices, unlabeled_indices))

        return indices

    @staticmethod
    def get_Wn(X: np.ndarray, k: int | tuple[int, int], gamma: float,
               semi_mask: np.ndarray=None,
               alpha: float=None) -> tuple[scipy.sparse.csr_matrix, np.ndarray, scipy.sparse.csr_matrix | None]:
        n = X.shape[0]  # 样本数。

        # -* 构建faiss索引器。
        index = LabelPropagator.build_faiss_index(X, semi_mask=semi_mask)

        # -* 获取最近邻。
        logger.info(f'获取最近邻...')
        a, I = index.search(X, k + 1 if isinstance(k, int) else k)  # +1是因为自身也会被搜索到。如果是分块搜索，就不纠结这个+1了。

        # -* 构造邻接矩阵。
        logger.info(f'构造邻接矩阵...')
        a = a[:, 1:] ** gamma  # 去掉自身，然后对距离做gamma调整。[N,K],f4.
        I = I[:, 1:]  # 去掉自身。[N,K],i4.

        row_idx = np.arange(n)  # 行索引。[N],i4.
        row_idx_rep = np.tile(row_idx, (k if isinstance(k, int) else (sum(k) - 1), 1)).T  # 行索引的重复。[N,K],i4.

        W = scipy.sparse.csr_matrix((a.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(n, n))
        W = W + W.T  # 无向图。
        W = W - scipy.sparse.diags(W.diagonal())  # 去掉自环。

        # -** 度归一化。
        S = W.sum(axis=1)  # degree。[N,1],f4.
        S[S == 0] = 1  # 防止除0。
        D = np.array(1. / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))  # 度矩阵的-0.5次方。[N,N],f4.
        Wn = D * W * D  # 归一化的邻接矩阵。[N,N],f4.

        # -* 计算系数矩阵。
        if alpha is not None:
            A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn  # LP公式 $A = I - alpha * Wn$.
        else:
            A = None

        return Wn, np.squeeze(np.asarray(S)), A

    @staticmethod
    def solve_lp(A: scipy.sparse.csr_matrix, Y: np.ndarray, max_iter: int, cg_guess: bool) -> np.ndarray:
        n, c = Y.shape
        Y_star = np.zeros((n, c))  # 初始化传播结果。[N,C],f4.

        for i in tqdm(range(c), desc='标签传播', ncols=c, unit='类', dynamic_ncols=True):
            y = Y[:, i]  # 获取第i类的标签。[N],f4.
            x0 = y if cg_guess else None
            f, cg_exit_code = scipy.sparse.linalg.cg(A, y, x0=x0, tol=1e-6, maxiter=max_iter)  # 求解LP公式。[N],f4.
            if cg_exit_code != 0:
                logger.warning(f'CG solver for class {i} did not converge. Exit code: {cg_exit_code}')
            Y_star[:, i] = f  # 保存第i类的传播结果。

        Y_star[Y_star < 0] = 0  # 修正负值。

        return Y_star

    @staticmethod
    def solve_lp_gpu(A: scipy.sparse.csr_matrix, Y: np.ndarray, max_iter: int, cg_guess: bool) -> np.ndarray:
        Y = cp.asarray(Y)
        A: c_scipy.sparse.csr_matrix = c_scipy.sparse.csr_matrix(A)

        n, c = Y.shape
        Y_star = cp.zeros((n, c))  # 初始化传播结果。[N,C],f4.

        for i in tqdm(range(c), desc='标签传播', ncols=c, unit='类', dynamic_ncols=True):
            y = Y[:, i]  # 获取第i类的标签。[N],f4.
            x0 = y if cg_guess else None
            f, cg_exit_code = c_linalg.cg(A, y, x0=x0, tol=1e-6, maxiter=max_iter)  # 求解LP公式。[N],f4.
            if cg_exit_code != 0:
                logger.warning(f'CG solver for class {i} did not converge. Exit code: {cg_exit_code}')
            Y_star[:, i] = f  # 保存第i类的传播结果。

        Y_star[Y_star < 0] = 0  # 修正负值。

        return Y_star.get()


    @dataclass
    class LPRet(object):
        Y_star: np.ndarray | None = None  # [N,C],f4，传播结果。
        degree: np.ndarray | None = None  # [N],f4，邻接矩阵（对称化且无自环）的度。
        # Wn: scipy.sparse.csr_matrix  # [N,N],f4，归一化的邻接矩阵。
        A: scipy.sparse.csr_matrix | None = None  # [N,N],f4，LP公式的系数矩阵。

    def __call__(self, X: np.ndarray, Y: np.ndarray, semi_mask: np.ndarray=None) -> LPRet:
        # -* 参数检查。
        if (not isinstance(self.k, int)) and self.k is not None:
            assert semi_mask is not None  # 分块搜索，需要semi_mask。

        # -* 获取形状信息。
        logger.info(f'{self} start: X.shape={X.shape}, Y.shape={Y.shape}')
        n = X.shape[0]  # 样本数。
        # c = Y.shape[1]  # 类别数。

        # -* 获取邻接矩阵。
        if self.A is None:
            _, degree, A = self.get_Wn(X, self.k, self.gamma,
                                       semi_mask=None if isinstance(self.k, int) else semi_mask,  # 若k为int，不分块则不传入semi_mask。
                                       alpha=self.alpha)
        else:
            degree, A = self.degree, self.A
            assert degree.shape == (n,) and A.shape == (n, n)

        # -* 若只需要A、degree。
        if self.A_degree_only:
            return self.LPRet(Y_star=None, degree=degree, A=A)

        # -* 标签传播。
        if self.cupy:
            Y_star = self.solve_lp_gpu(A, Y, self.max_iter, self.cg_guess)
        else:
            Y_star = self.solve_lp(A, Y, self.max_iter, self.cg_guess)

        return self.LPRet(Y_star=Y_star, degree=degree, A=A)

    def lp_packet_cal(self, lp_packet: 'LPPacket'):
        lp_ret = self(lp_packet.X, lp_packet.Y, semi_mask=lp_packet.semi_mask)
        lp_packet.lp_score, lp_packet.degree, lp_packet.A = lp_ret.Y_star, lp_ret.degree, lp_ret.A
        return lp_packet

    # -* 以下为X预处理方法。

    @staticmethod
    def X_preprocess(X: np.ndarray, X_method: dict[str, ...]=None) -> np.ndarray:
        match X_method:
            case None | {'method': 'identical'}:
                return X
            case {'method': 'L2'}:
                return prep.normalize(X, norm='l2')  # [N,d],f4
            case _:
                raise ValueError(f"Unknown X_method: {X_method}")

    # -* 以下为Y预处理方法。

    @staticmethod
    def cls_balance(score: np.ndarray) -> np.ndarray:
        """做类别平衡。

        Args:
            score: [N,K],f4，得分。

        Returns: [N,K],f4，Y
        """
        cls_num = score.shape[1]
        cls_score = score.sum(axis=0)  # [K],f4，每个类别的总得分。
        total_score = cls_score.sum()

        cls_weight = total_score / (cls_num * cls_score + 1e-8)  # [K],f4，每个类别的权重。

        return score * cls_weight[None, :]

    @staticmethod
    def bg_fg_balance(score: np.ndarray, balanced_ratio: str='eq') -> np.ndarray:
        """做背景类别与前景类别平衡。

        Args:
            score: [N,K],f4，得分。
            balanced_ratio: str，平衡比例。'eq'表示平衡后前背景得分相等，'1/F'表示平衡后前背景得分比例为1/前景类别数。

        Returns: [N,K],f4，Y
        """
        cls_num = score.shape[1]
        bg_score, fg_score = score[:, 0].sum(), score[:, 1:].sum()  # 背景得分，前景得分。
        total_score = bg_score + fg_score

        match balanced_ratio:
            case 'eq':
                bg_weight = total_score / (2 * bg_score + 1e-8)
                fg_weight = total_score / (2 * fg_score + 1e-8)
            case '1F':
                bg_weight = total_score / (cls_num * bg_score + 1e-8)
                fg_weight = ((cls_num - 1) * total_score) / (cls_num * fg_score + 1e-8)
            case _:
                raise ValueError(f"Unknown balanced_ratio: {balanced_ratio}")

        Y = score.copy()
        Y[:, 0] *= bg_weight
        Y[:, 1:] *= fg_weight

        return Y

    @staticmethod
    def alpha_bg(score: np.ndarray, alpha: float) -> np.ndarray:
        """将背景类别的得分进行alpha次幂。

        Args:
            score: [N,K],f4，得分。
            alpha: float，幂次。

        Returns: [N,K],f4，Y
        """
        Y = score.copy()
        Y[:, 0] **= alpha
        return Y

    @staticmethod
    def semi_balance(score: np.ndarray, semi_mask: np.ndarray, balanced_ratio: float=1.) -> np.ndarray:
        """做半监督平衡。

        Args:
            score: [N,K],f4，得分。
            semi_mask: [N],bool，True表示labeled。
            balanced_ratio: str，平衡比例。labeled_score与unlabeled_score的比例。

        Returns: [N,K],f4，Y
        """
        labeled_score, unlabeled_score = score[semi_mask, :].sum(), score[~semi_mask, :].sum()
        total_score = labeled_score + unlabeled_score

        labeled_weight = (balanced_ratio / (1 + balanced_ratio)) * total_score / (labeled_score + 1e-8)
        unlabeled_weight = (1 / (1 + balanced_ratio)) * total_score / (unlabeled_score + 1e-8)

        Y = score.copy()
        Y[semi_mask, :] *= labeled_weight
        Y[~semi_mask, :] *= unlabeled_weight

        return Y
