#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/19 21:54
@File    : grad_scaler_manager.py
@Software: PyCharm
@Desc    : 
"""
from typing import Union, Iterable, Optional, Tuple, Dict, List

import torch
from alchemy_cat.py_tools import ADict
from torch.utils._foreach_utils import _has_foreach_support, _group_tensors_by_device_and_dtype

__all__ = ['GradScalerManager', 'get_grad_norm']


class GradScalerManager(object):
    """给定loss和optimizer，使用GradScaler进行后向、梯度裁剪、更新、清理，并返回中间结果。"""
    state_dict_key = "amp_scaler"

    def __init__(self,
                 scaler: torch.cuda.amp.GradScaler,
                 optimizer: torch.optim.Optimizer,
                 grad_max_norm: float=None):
        """
        Initializes the GradScalerManager with a GradScaler.

        Args:
            scaler (torch.cuda.amp.GradScaler): The GradScaler to be managed.
            optimizer (torch.optim.Optimizer): The optimizer to be updated.
            grad_max_norm (float, optional): Maximum allowed norm for the gradients.
                If None, no clipping is done.
        """
        self.scaler = scaler
        self.optimizer = optimizer
        self.grad_max_norm = grad_max_norm

    def __call__(self, loss: torch.Tensor,
                 retain_graph: bool=False, create_graph: bool=False,
                 update_grad=True,
                 zero_grad: bool=True) -> ADict:
        """
        This method is called to scale the loss, perform backward pass, unscale and clip gradients,
        and update the optimizer.

        Args:
            loss (torch.Tensor): The loss tensor to be scaled and back-propagated.
            retain_graph (bool, optional): Whether to retain the computation graph after backward pass.
                Default is False.
            create_graph (bool, optional): Whether to create a computation graph from the backward pass.
                Default is False.
            update_grad (bool, optional): Whether to update the gradients. Default is True.
            zero_grad (bool, optional): Whether to zero the gradients after update. Default is True.

        Returns:
            ADict: A dictionary with the updated gradient norm, scale,
                and a flag indicating if the update was invalid.
        """
        ret = ADict()

        # -* 缩放损失并后向。
        self.scaler.scale(loss).backward(retain_graph=retain_graph, create_graph=create_graph)

        # -* 如果做梯度更新（有时多次backward后，不需要更新梯度），缩放、裁剪、更新梯度。
        if update_grad:
            # -** 缩放梯度。
            self.scaler.unscale_(self.optimizer)
            # -** 裁剪梯度。
            parameters = [p  # 找到所有会被optimizer“有效”更新的参数。
                          for group in self.optimizer.param_groups
                          for p in group['params']
                          if p.grad is not None and p.requires_grad]
            if self.grad_max_norm is not None:
                ret.grad_norm = torch.nn.utils.clip_grad_norm_(parameters,
                                                               max_norm=self.grad_max_norm, error_if_nonfinite=False)
            else:
                ret.grad_norm = get_grad_norm(parameters)
            # -** 更新梯度。
            self.scaler.step(self.optimizer)
            # -** 更新缩放器。
            ret.scale_before_update = self.scaler.get_scale()
            self.scaler.update()
            ret.scale = self.scaler.get_scale()
            ret.invalid_update = ret.scale_before_update > ret.scale  # 若缩放因子变小，说明此前更新无效。
            # -** 清空梯度。
            if zero_grad:
                self.optimizer.zero_grad(set_to_none=True)
        else:
            ret.grad_norm = None
            ret.scale_before_update = None
            ret.scale = None
            ret.invalid_update = None

        return ret

    def state_dict(self) -> dict[str, ...]:
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, ...]):
        self.scaler.load_state_dict(state_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}(scaler={self.scaler}, grad_max_norm={self.grad_max_norm})"


def get_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], norm_type: float = 2.0,
        foreach: Optional[bool] = None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)

    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[torch.Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]

    if norm_type == torch.inf:
        norms = [g.detach().abs().max().to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), [grads]) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.norm(g, norm_type) for g in grads])

        total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm
