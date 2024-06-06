#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/19 13:17
@File    : emb_classifier.py
@Software: PyCharm
@Desc    : 
"""
import os
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from alchemy_cat.py_tools import set_rand_seed, hash_str, ADict
from alchemy_cat.torch_tools import update_model_state_dict

from libs.classifier.typing import SPEmbClassifier

__all__ = ['EmbLinearProbe', 'EmbMLP', 'MLP', 'EmbClassifier']

EmbClassifier = t.TypeVar("EmbClassifier", bound="EmbLinearProbe")


class EmbLinearProbe(nn.Module):
    def __init__(self, emb_dim: int, cls_num: int, scale: float, bias: bool=False, cos_sim: bool=True):
        """
        embedding线性分类器。
        Args:
            emb_dim: embedding维度。
            cls_num: 类别数。
            scale: logits缩放因子。
            bias: 是否使用bias。
            cos_sim: 是否使用cosine similarity计算logits。
        """

        super().__init__()

        self.emb_dim = emb_dim
        self.cls_num = cls_num
        self.bias = bias
        self.cos_sim = cos_sim

        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

        # -* 定义投影层。
        self.proj = nn.Identity()

        # -* 定义权重矩阵
        self.head = nn.Linear(emb_dim, cls_num, bias=bias)

    def forward(self, emb: torch.Tensor) -> ADict:
        """前向计算得到缩放后的logits。"""
        out = ADict()
        # -* 数据预处理。
        emb = emb.to(device=next(self.parameters()).device, non_blocking=True)

        # -* 通过投影层。
        emb = self.proj(emb)

        # -* 计算相似度.
        if self.cos_sim:
            emb_norm = F.normalize(emb, p=2, dim=-1)
            weight_norm = F.normalize(self.head.weight, p=2, dim=-1)
            sim = emb_norm @ weight_norm.t()
            if self.head.bias is not None:
                sim = sim + self.head.bias
        else:
            sim = self.head(emb)

        # -* 缩放logits.
        logits = sim * self.scale

        # -* 构造输出。
        out.logits = logits
        return out

    def load_ckp(self, ckp_pth: str | os.PathLike):
        """加载checkpoint。"""
        state_dict = torch.load(ckp_pth)['state_dict']
        update_model_state_dict(self, state_dict, verbosity=3)

    def initialize(self, seed: int):
        """初始化权重。"""
        set_rand_seed(hash_str('linear_probe_head') + seed)  # 设置linear probe组的随机种子。
        self.head.reset_parameters()
        if self.head.bias is not None:
            self.head.bias.data.zero_()  # 对分类头，bias初始化为0。

    def emb_dt_inp_cal(self, inp: ADict) -> ADict:
        """emb分类器的输入数据处理。"""
        return self(inp.emb)

    def classify_samq_emb_cal(self, emb: torch.Tensor, _: str) -> torch.Tensor:
        """emb分类器的输入数据处理。"""
        return self(emb).logits

    @classmethod
    def classify_samq_emb_val_classifier(cls: type(EmbClassifier),
                                         ckp_pth: str | os.PathLike,
                                         emb_dim: int, cls_num: int, scale: float,
                                         bias: bool=False, cos_sim: bool=True,
                                         **kwargs) -> SPEmbClassifier:
        """得到classify_samq_emb中的分类器。"""
        classifier = cls(emb_dim, cls_num, scale, bias, cos_sim, **kwargs)
        classifier.load_ckp(ckp_pth)
        classifier.eval().to('cuda')

        return classifier.classify_samq_emb_cal

    def get_pg(self, lr: float, weight_decay: float) -> list[dict[str, ...]]:
        """获取优化器参数组。"""
        pg = [{'pg_name': 'head_weight', 'params': [self.head.weight], 'lr': lr, 'weight_decay': weight_decay}]
        if self.head.bias is not None:
            pg.append({'pg_name': 'head_bias', 'params': [self.head.bias], 'lr': lr, 'weight_decay': 0.})

        return pg


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class EmbMLP(EmbLinearProbe):

    def __init__(self, emb_dim: int, cls_num: int, scale: float, bias: bool=False, cos_sim: bool=True,
                 num_layers: int=2, hidden_factor: int=4):
        """
        EmbMLP is a subclass of EmbLinearProbe. It is a multi-layer perceptron (MLP) model for embedding classification.

        Args:
            emb_dim (int): The dimension of the embedding.
            cls_num (int): The number of classes.
            scale (float): The scaling factor for logits.
            bias (bool, optional): Whether to use bias. Defaults to False.
            cos_sim (bool, optional): Whether to use cosine similarity for calculating logits. Defaults to True.
            num_layers (int, optional): The number of layers in the MLP. Defaults to 2.
            hidden_factor (int, optional): The factor for the hidden dimension size in the MLP. Defaults to 4.
            """

        super().__init__(emb_dim, cls_num, scale, bias, cos_sim)

        self.num_layers = num_layers
        self.hidden_factor = hidden_factor

        # -* 定义投影层。
        self.proj = MLP(emb_dim, emb_dim * hidden_factor, emb_dim, num_layers)

    def initialize(self, seed: int):
        """初始化权重。"""
        super().initialize(seed)
        set_rand_seed(hash_str('mlp') + seed)  # 设置MLP组的随机种子。
        for layer in self.proj.layers:
            layer.reset_parameters()

    def get_pg(self, lr: float, weight_decay: float) -> list[dict[str, ...]]:
        """获取优化器参数组。"""
        weights = [l.weight for l in self.modules() if isinstance(l, nn.Linear)]
        biases = [l.bias for l in self.modules() if isinstance(l, nn.Linear) and l.bias is not None]

        pg = [{'pg_name': 'weights', 'params': weights, 'lr': lr, 'weight_decay': weight_decay},
              {'pg_name': 'biases', 'params': biases, 'lr': lr, 'weight_decay': 0.}]

        return pg
