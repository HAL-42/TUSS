#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/19 21:16
@File    : emb_classify_loss.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn.functional as F
from alchemy_cat.py_tools import ADict
from torch import nn

__all__ = ['EmbClassifyLoss']


class EmbClassifyLoss(nn.Module):

    def __init__(self, weight: torch.Tensor=None, sample_weighted: bool | dict[str, ...]=True):
        """
        embedding分类器的loss。
        Args:
            sample_weighted: 是否对样本加权。
        """
        super().__init__()
        self.sample_weighted = sample_weighted

        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                sample_weight: torch.Tensor=None, conf: torch.Tensor=None) -> torch.Tensor:
        """计算loss。
        Args:
            logits: [N, C]模型输出的logits。
            labels: [N, C]样本概率。
            sample_weight: [N]样本权重。
            conf: [N]样本置信度。
        """
        # -* 检查、处理输入。
        labels = labels.to(device=logits.device, non_blocking=True)
        if sample_weight is not None:
            sample_weight = sample_weight.to(device=logits.device, non_blocking=True).to(dtype=logits.dtype)
        if conf is not None:
            conf = conf.to(device=logits.device, non_blocking=True).to(dtype=logits.dtype)

        # -* 权重预处理。
        match self.sample_weighted:
            case True | {'method': 'L1'}:
                w = sample_weight / (sample_weight.sum() + 1e-3)
            case False | {'method': 'no_weight'}:
                w = None
            case {'softmax': float(gamma)}:
                w = torch.softmax(sample_weight * gamma, dim=0)
            case {'L1_gamma_L1': float(gamma)}:
                w = F.normalize(F.normalize(sample_weight, p=1, dim=0) ** gamma, p=1, dim=0)
            case _:
                raise ValueError(f"Unknown sample_weighted: {self.sample_weighted}")

        # -* 计算loss。
        loss = self.cross_entropy(logits, labels)

        # -* 作用置信度。
        if conf is not None:
            loss = loss * conf

        # -* 作用权重。
        if w is None:
            loss = loss.mean()
        else:
            loss = (loss * w).sum()

        return loss

    def emb_dt_emb_classifier_cal(self, inp: ADict, out: ADict):
        """计算embedding分类器的loss。
        Args:
            inp: 输入。
            out: 输出。
        """
        return self(out.logits, inp.soft_lb, inp.area, inp.conf if 'conf' in inp else None)
