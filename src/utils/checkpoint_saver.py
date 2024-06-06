#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/18 20:52
@File    : checkpoint_saver.py
@Software: PyCharm
@Desc    : 
"""
import os
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from alchemy_cat.py_tools import file_md5
from timm.utils import CheckpointSaver as TimmCheckpointSaver, unwrap_model, get_state_dict
from torch import nn
from torch.optim import Optimizer

__all__ = ['CheckpointInfo', 'CheckpointSaver']


@dataclass(eq=True, order=True, slots=True, kw_only=True)
class CheckpointInfo(object):
    """Checkpoint信息。"""
    metric: t.Any = None  # 都有metric时比metric，没有metric的，设置为最糟情况。
    epoch_for_cmp: int | float = None  # 用于比较的epoch，metric相等时，epoch大的优先。
    epoch: int | float = field(default=None, compare=False)
    save_path: str = field(default=None, compare=False)
    decreasing: bool = field(default=False, compare=False)

    def __post_init__(self):
        if self.epoch is None:
            raise ValueError("epoch can't be None.")
        if self.save_path is None:
            raise ValueError("save_path can't be None.")
        if self.epoch_for_cmp is not None:
            raise ValueError("epoch_for_cmp can't be set.")

        if self.metric is None:
            self.metric = float('-inf') if not self.decreasing else float('inf')

        self.epoch_for_cmp = self.epoch if not self.decreasing else -self.epoch


class CheckpointSaver(TimmCheckpointSaver):

    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer=None,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix: str='iter',
            recovery_prefix: str='iter',
            checkpoint_dir: str | os.PathLike='/tmp/checkpoints',
            recovery_dir: str | os.PathLike='/tmp/recovery',
            small_is_better: bool=False,
            max_history: int | float=float('inf'),
            unwrap_fn: t.Callable[[nn.Module], nn.Module]=unwrap_model):

        super().__init__(model, optimizer, args, model_ema, amp_scaler, checkpoint_prefix, recovery_prefix,
                         str(checkpoint_dir), str(recovery_dir), small_is_better, max_history, unwrap_fn)

        # -* 覆盖TimmCheckpointSaver中的状态量。
        self.checkpoint_files: list[CheckpointInfo] = []

        # -* 覆盖TimmCheckpointSaver中的默认配置。
        self.extension = '.pth'

    @property
    def worst_file(self) -> CheckpointInfo | None:
        return self.checkpoint_files[-1] if self.checkpoint_files else None

    @property
    def best_file(self) -> CheckpointInfo | None:
        return self.checkpoint_files[0] if self.checkpoint_files else None

    def cache_checkpoint(self, epoch: int, metric=None) -> str:
        """缓存当前epoch的模型，返回缓存文件路径。"""
        assert epoch >= 0

        # -* 缓存当前epoch的模型；先保存到临时文件，防止中断时保存的文件不完整。
        tmp_cache_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_cache_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_cache_path, epoch, metric)
        if os.path.exists(last_cache_path):
            os.unlink(last_cache_path)  # required for Windows support.
        os.rename(tmp_cache_path, last_cache_path)
        logger.info(f"CACHE checkpoint with metric={metric} at epoch-{epoch} to {last_cache_path}. \n"
                     f"File MD5: {file_md5(last_cache_path)}")

        return last_cache_path

    def save_checkpoint(self, epoch: int | float, metric=None, last_cache_path: str=None) -> CheckpointInfo | None:
        """相比TimmCheckpointSaver:
        1. 增加更详细的输出。
        2. metric为None时，按照epoch排序。
        """
        assert epoch >= 0
        epoch_name = 'final' if epoch == float('inf') else str(epoch)

        # -* 构造当前epoch的CheckpointInfo。
        last_file = CheckpointInfo(metric=metric,
                                   epoch=epoch,
                                   save_path=os.path.join(self.checkpoint_dir,
                                                          '-'.join([self.save_prefix, epoch_name]) + self.extension),
                                   decreasing=self.decreasing)
        # -** 检查是否有重复的epoch。
        for file in self.checkpoint_files:
            if file.epoch == last_file.epoch:
                raise ValueError(f"epoch={epoch} has already been saved to {file.save_path}.")

        # -* 若last_cache_path为None，则缓存当前epoch的模型。
        if last_cache_path is None:
            last_cache_path = self.cache_checkpoint(epoch, metric)
        # -* 反之读取之前的cache，修正metric。
        else:
            save_state = torch.load(last_cache_path, map_location='cpu')
            save_state['epoch'] = epoch
            save_state['metric'] = metric
            torch.save(save_state, last_cache_path)

        # -* 判断是否需要正式保存当前epoch的模型。
        if (len(self.checkpoint_files) < self.max_history  # 没有达到最大保存数。
                or self.cmp(last_file, self.worst_file)):  # 若达到最大保存数目，则worst_file一定存在。比较当前是否优于此前最差。
            # -* 清理多余的checkpoint。
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            # -* 正式保存当前epoch的模型。
            if Path(last_file.save_path).exists():
                Path(last_file.save_path).unlink()
            os.link(last_cache_path, last_file.save_path)

            # -* 追踪当前epoch的CheckpointInfo。
            self.checkpoint_files.append(last_file)
            self.checkpoint_files = sorted(
                self.checkpoint_files,
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            # -* 更新最佳信息。
            if self.best_epoch != self.best_file.epoch:
                self.best_epoch = self.best_file.epoch
                self.best_metric = self.best_file.metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_cache_path, best_save_path)

            checkpoints_str = (f"SAVE last checkpoint with metric={metric} at epoch={epoch_name} "
                               f"to {last_file.save_path}. \n"
                               f"File MD5: {file_md5(last_file.save_path)} \n"
                               f"Current has [{len(self.checkpoint_files)}/{self.max_history}] checkpoints. \n"
                               f"Best metric={self.best_file.metric} at epoch={self.best_file.epoch}. \n"
                               f"Worst metric={self.worst_file.metric} at epoch={self.worst_file.epoch}. \n")
            logger.success(checkpoints_str)
        else:
            checkpoints_str = (f"NOT SAVE last checkpoint with metric={metric} at epoch={epoch_name}. \n"
                               f"Current has [{len(self.checkpoint_files)}/{self.max_history}] checkpoints. \n"
                               f"Best metric={self.best_file.metric} at epoch={self.best_file.epoch}. \n"
                               f"Worst metric={self.worst_file.metric} at epoch={self.worst_file.epoch}. \n")
            logger.success(checkpoints_str)

        return self.best_file

    def _save(self, save_path, epoch, metric=None):
        """相比TimmCheckpointSaver，支持optimizer为None。"""
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__,
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'metric': metric,
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.optimizer is not None:
            save_state['optimizer'] = self.optimizer.state_dict()
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(save_state, save_path)
