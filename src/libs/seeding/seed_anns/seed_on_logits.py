#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/24 22:11
@File    : score_cw.py
@Software: PyCharm
@Desc    : 
"""
import typing as t
import warnings

import torch
import torch.nn.functional as F
from alchemy_cat.contrib.tasks.wsss.seeding2 import cam2score_cuda, cat_bg_score_cuda, idx2seed_cuda

from libs.sam import SamAnns
from .utils import scatter_anns, sort_anns

__all__ = ['seed_on_logits']


def seed_on_logits(sps: SamAnns, sp_logits: torch.Tensor, fg_cls: torch.Tensor,
                   with_bg_logit: bool=False,
                   drop_bg_logit: bool=False,
                   fg_method: str | dict=None,
                   bg_method: str | dict=None,
                   priority: str | tuple[str, ...]=None,
                   ret_seeded_sps: bool=False) -> torch.Tensor | tuple[torch.Tensor, SamAnns]:
    """先收集标注的CAM响应，再在标注间做归一化，计算背景得分，channel维度上argmax后得到种子点。

    Args:
        sps: SAM标注，应该已经完成stack_data。
        sp_logits: (S, K+B)的分类logits。B指dust bin。
        fg_cls: (F,) 的前景类别标签。
        with_bg_logit: 是否包含背景logits。若包含，则默认第0个为背景logits，且背景类总是存在。fg_method负责前背景分计算，
            bg_method置为`{'method': 'no_bg'}`.
        drop_bg_logit: 是否丢弃背景logits。
        fg_method: 前景分计算配置。
        bg_method: 背景分计算配置。
        priority: 标注排序优先级。
        ret_seeded_sps: 是否返回排序后，带有置信度得分和种子的sp。

    Returns:
        (H, W) 种子点。
    """
    if drop_bg_logit:
        assert not with_bg_logit, "drop_bg_logit and with_bg_logit can't be True at the same time."
        sp_logits = sp_logits[:, 1:]  # (S, K)

    if with_bg_logit and bg_method is None:  # 若未指定背景计算方法。
        bg_method = {'method': 'no_bg'}  # 默认无需计算背景。

    sp_logits = sp_logits.T[:, None, :]  # (K+B, 1, S)，改变形状适配CAM算子——一般输入为(C, H, W)。

    match fg_method:
        case {'softmax': float() | None as gamma, 'norm': is_norm, **others}:
            L1, bypath_suppress = others.get('L1', False), others.get('bypath_suppress', True)
            is_score = others.get('is_score', False)
            fg_suppress = others.get('fg_suppress', None)

            if L1 and (gamma is not None) and (not is_score):
                warnings.warn("L1 is True, but is_score is False.")

            def norm(x: torch.Tensor) -> torch.Tensor:  # x: (K+B或F, 1, S)
                if L1:
                    x = F.normalize(x, p=1, dim=0)

                if gamma is not None:
                    if is_score:
                        x = x.clone()
                        x[x < 1e-8] = -torch.inf  # 若为score，将小于1e-8的值置为-inf，防止干扰softmax。
                    return torch.softmax(x * gamma, dim=0)
                else:
                    return x

            if with_bg_logit:
                score_idx = F.pad(t.cast(torch.Tensor, fg_cls + 1), (1, 0), mode='constant', value=0)  # (F+1,)
            else:
                score_idx = fg_cls  # (F,)

            if bypath_suppress:
                sp_raw_score = norm(sp_logits)  # (K+B, 1, S)
                fg_score = sp_raw_score[score_idx, :, :]  # (F, 1, S) / (F+1, 1, S)
            else:
                fg_logits = sp_logits[score_idx, :, :]  # (F, 1, S) / (F+1, 1, S)
                fg_score = norm(fg_logits)  # (F, 1, S) / (F+1, 1, S)

            match fg_suppress:
                case None:
                    pass
                case {'thresh': float(fg_suppress_thresh), 'keep_bg': bool(keep_bg)}:
                    fg_max_score = torch.max(fg_score[:, 0, :], dim=1)[0]  # (F,) / (F+1,)
                    suppress_mask = t.cast(torch.Tensor, fg_max_score < fg_suppress_thresh)
                    if keep_bg:
                        suppress_mask[0] = False
                    fg_score[suppress_mask, :, :] = -torch.inf
                case _:
                    raise ValueError(f"Unknown fg_suppress: {fg_suppress}")

            if is_norm:
                fg_score = cam2score_cuda(fg_score, dsize=None, resize_first=True)  # (F, 1, S) / (F+1, 1, S)
        case _:
            raise ValueError(f"Unknown fg_method: {fg_method}")

    # -* 计算标注背景得分。
    sp_score = cat_bg_score_cuda(fg_score, bg_method)  # (F+1, 1, S)
    if ret_seeded_sps:
        sps.add_item('score', sp_score[:, 0, :].T)

    sp_max_idx = torch.argmax(sp_score, dim=0)  # (1, S)

    # -* 得到标注种子点。
    sp_seed = idx2seed_cuda(sp_max_idx, fg_cls)  # (1, S)
    sps.add_item('seed', sp_seed[0].tolist())

    # -* 以得分为标注置信度。
    sp_conf = torch.gather(sp_score, dim=0, index=sp_max_idx[None, ...])[0]  # (1, S)，与max效果相同。
    # sp_conf = sp_score[sp_max_idx,
    # torch.arange(sp_score.shape[1])[:, None],
    # torch.arange(sp_score.shape[2])[None, :]]  # (1, S)，与max效果相同。
    sps.add_item('conf', sp_conf[0].tolist())

    # -* 对标注排序。
    sorted_sps = sort_anns(sps, priority=priority)

    # -* 得到最终种子点。
    sorted_sps_seed = torch.as_tensor([sp['seed'] for sp in sorted_sps],
                                       dtype=sp_seed.dtype, device=sp_seed.device)  # (S,)
    seed = scatter_anns(sorted_sps_seed, sorted_sps, default_vec=0)

    if not ret_seeded_sps:
        return seed
    else:
        return seed, sps
