#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/2 23:12
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pp
from types import FunctionType

import torch
from alchemy_cat.data import inf_loader
from alchemy_cat.py_tools import meow, set_rand_seed, Config, \
    ADict
from alchemy_cat.torch_tools import init_env, MovingAverageValueTracker, cal_loss_items, ValidationEnv
from loguru import logger
from torch import nn
from torch.cuda import amp
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from utils.grad_scaler_manager import GradScalerManager
from utils.checkpoint_saver import CheckpointSaver
from tasks.classify_samq_emb import run as classify_samq_emb_run


def val(val_cfg: Config, model: nn.Module, model_val_cal: FunctionType) -> ADict:
    logger.info("================================== Validation ==================================")
    val_cfg.parse(experiments_root='').compute_item_lazy().freeze()
    pp(val_cfg)

    with ValidationEnv(model, change_benchmark=False, seed=0, empty_cache=True, cv2_num_threads=-1) as val_model:
        classifier = model_val_cal.__get__(val_model)
        val_rslt = classify_samq_emb_run.main(argparse.Namespace(), val_cfg, classifier=classifier)

    logger.info("================================ Validation End ================================")
    return val_rslt


def main(args: argparse.Namespace, cfg: str | Config) -> ADict:
    # -* 数据集。
    dt = cfg.dt.cls(**cfg.dt.ini)
    print(dt, end="\n\n")

    # -* 训练数据增强器。
    auger = cfg.auger.cls(dt, **cfg.auger.ini)
    print(auger, end="\n\n")

    # -* 数据加载器。
    sampler = cfg.sampler.cls(auger, **cfg.sampler.ini,
                              generator=torch.Generator().manual_seed(cfg.rand_seed + cfg.rand.data))
    loader = DataLoader(auger,
                        batch_size=cfg.loader.batch_size,
                        sampler=sampler,
                        num_workers=cfg.loader.num_workers,
                        pin_memory=bool(args.pin_memory),
                        drop_last=True,
                        generator=torch.Generator().manual_seed(cfg.rand_seed + cfg.rand.data),
                        prefetch_factor=args.prefetch_factor if cfg.loader.num_workers > 0 else None,
                        persistent_workers=True if cfg.loader.num_workers > 0 else False
                        )
    inf_train_loader = inf_loader(loader)
    print(sampler)
    print(loader, end="\n\n")

    # -* 分类模型。
    model: nn.Module = cfg.model.cls(emb_dim=dt.emb_dim, cls_num=dt.cls_num, **cfg.model.ini)
    model: torch.nn.Module

    model.initialize(seed=(model_ini_seed := cfg.rand_seed + cfg.rand.model))
    print(f"以 {model_ini_seed} 为随机种子初始化模型。")

    model.train().to('cuda')
    print(model, end="\n\n")

    # -* 构造损失函数。
    loss_items = cfg.loss.loss_items.branch_copy()
    for item_name, loss_item in loss_items.items():
        if 'cls' in loss_item:
            loss_item.cri = loss_item.cls(**loss_item.ini)
    print(loss_items, end="\n\n")

    # -* 优化器。
    pg = cfg.opt.get_pg.cal(model, **cfg.opt.get_pg.ini)
    opt = cfg.opt.cls(params=pg, lr=0., **cfg.opt.ini)  # lr在组内设置，其余默认为0。
    print(pg)
    print(opt, end="\n\n")

    # -* 调整学习率。
    warm_sched = cfg.sched.warm.cls(opt, **cfg.sched.warm.ini)
    main_sched = cfg.sched.main.cls(opt, **cfg.sched.main.ini)
    sched = SequentialLR(opt, [warm_sched, main_sched], [cfg.sched.warm.warm_iters])
    print(warm_sched)
    print(main_sched)
    print(sched, end="\n\n")

    # -* 损失放大器。
    scaler_manager = GradScalerManager(amp.GradScaler(**cfg.amp.scaler.ini), opt,
                                       grad_max_norm=cfg.opt.grad_clip.grad_max_norm)
    print(scaler_manager, end="\n\n")

    # -* 构造模型保存器，并保存初始模型。
    ckp_saver = CheckpointSaver(model,
                                checkpoint_prefix='iter',
                                checkpoint_dir=Path(cfg.rslt_dir) / 'checkpoints',
                                small_is_better=False,
                                max_history=float('inf'))
    ckp_saver.save_checkpoint(epoch=0, metric=None)

    # -* 跟踪器。
    loss_trackers = defaultdict(lambda: MovingAverageValueTracker(cfg.solver.loss_average_step *  # 窗长要乘以子迭代次数。
                                                                  cfg.solver.sub_iter_num))
    print(loss_trackers, end="\n\n")

    # -* Tensorboard。
    meow.writer = writer = SummaryWriter(osp.join(cfg.rslt_dir, 'summary'), purge_step=0)

    for iteration in tqdm(range(cfg.solver.max_iter), dynamic_ncols=True,
                          desc='训练', unit='批次', miniters=cfg.solver.display_step,
                          bar_format='{l_bar}{bar}{r_bar}\n'):
        meow.iteration = iteration  # 在全局变量中记录当前迭代次数。

        # -* 设置此iteration的随机种子。
        set_rand_seed(iteration + cfg.rand_seed + cfg.rand.train)

        for sub_iteration in range(cfg.solver.sub_iter_num):  # 遍历所有子迭代。
            # -* 获取新一个批次数据。
            inp = next(inf_train_loader)

            with amp.autocast(enabled=cfg.amp.enabled):
                # -* 前向。
                out = cfg.model.cal(model, inp)

                # -* 计算损失。
                losses = cal_loss_items(loss_items, inp, out)

                # -* 记录损失。
                for loss_name, loss_val in losses.items():  # 记录没有平均过的子迭代损失，配合扩展后的窗长，可得正确结果。
                    loss_trackers[loss_name].update(loss_val.item())

                # -* 后向。
                scale_info = scaler_manager(losses['total_loss'] / cfg.solver.sub_iter_num,
                                            update_grad=(sub_iteration == cfg.solver.sub_iter_num - 1),
                                            zero_grad=True)

                if scale_info.invalid_update:  # 若缩放因子变小，说明此前更新无效。
                    logger.warning(f"当前scale={scale_info.scale_before_update}导致溢出，该次迭代无效。")

        # -* 调整学习率。
        sched.step()

        # -* writer记录当前值。
        # -** 损失。
        for loss_name, tracker in loss_trackers.items():
            writer.add_scalar(f'loss/{loss_name}', tracker.last, iteration + 1)
        # -** 学习率
        for group in opt.param_groups:
            writer.add_scalar(f'optim/lr/{group["pg_name"]}', group['lr'], iteration + 1)
        # -** 损失缩放因子。
        if scaler_manager.scaler.is_enabled():
            writer.add_scalar('optim/scale', scale_info.scale, iteration + 1)
        # -** 梯度范数。
        writer.add_scalar('optim/grad_norm', scale_info.grad_norm.item(), iteration + 1)

        # -* msg打印平均值。
        if (iteration + 1) % cfg.solver.display_step == 0:
            msg = f"[{iteration + 1}/{cfg.solver.max_iter}]: \n"
            # -** 损失。
            msg += "损失：\n"
            for loss_name, tracker in loss_trackers.items():
                msg += f"    {loss_name}: {tracker.mean}\n"
            logger.info(msg)

        # -* 验证性能。
        # 验证与最终推理是一样的，因此max_iter时不再验证、保存。
        if (iteration + 1) % cfg.solver.val_step == 0 and (iteration + 1) != cfg.solver.max_iter:
            (val_cfg := cfg.val.cfg.branch_copy()).rslt_dir = osp.join(cfg.rslt_dir, 'val', f'iter-{iteration + 1}')
            val_rslt = val(val_cfg, model, cfg.model.val_cal)  # noqa
            writer.add_scalar(f"val/seg/mIoU", metric := val_rslt.seg_metric.mIoU, iteration + 1)
            writer.add_scalar(f"val/seg/precision", val_rslt.seg_metric.macro_avg_precision, iteration + 1)
            writer.add_scalar(f"val/seg/recall", val_rslt.seg_metric.macro_avg_recall, iteration + 1)
            writer.add_scalar(f"val/cls/F1_score", val_rslt.cls_metric.F1_score, iteration + 1)
        else:
            metric = None

        # -* 保存训练中间模型。
        if (iteration + 1) % cfg.solver.save_step == 0 and (iteration + 1) != cfg.solver.max_iter:
            ckp_saver.save_checkpoint(epoch=iteration + 1, metric=metric)

    # -* 推理最终模型。
    (val_cfg := cfg.val.cfg.branch_copy()).rslt_dir = osp.join(cfg.rslt_dir, 'val', f'iter-final')
    val_rslt = val(val_cfg, model, cfg.model.val_cal)  # noqa
    writer.add_scalar(f"val/seg/mIoU", metric := val_rslt.seg_metric.mIoU, cfg.solver.max_iter)
    writer.add_scalar(f"val/seg/precision", val_rslt.seg_metric.macro_avg_precision, cfg.solver.max_iter)
    writer.add_scalar(f"val/seg/recall", val_rslt.seg_metric.macro_avg_recall, cfg.solver.max_iter)
    writer.add_scalar(f"val/cls/F1_score", val_rslt.cls_metric.F1_score, cfg.solver.max_iter)

    # -* 关闭Tensorboard，保存最终模型。
    ckp_saver.save_checkpoint(epoch=float('inf'), metric=metric)
    writer.close()

    return val_rslt


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-f', '--prefetch_factor', default=2, type=int)
    parser.add_argument('-p', '--pin_memory', default=0, type=int)
    parser.add_argument("-b", '--benchmark', default=0, type=int)
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    parser.add_argument("-e", '--eval_only', default=0, type=int)
    return parser


if __name__ == "__main__":
    # -* 读取命令行参数。
    _args = get_parser().parse_args()

    # -* 初始化环境。
    _, _cfg = init_env(is_cuda=True,
                       is_benchmark=bool(_args.benchmark),
                       is_train=True,
                       config_path=_args.config,
                       experiments_root="experiment",
                       rand_seed=True,
                       cv2_num_threads=0,
                       verbosity=True,
                       log_stdout=True,
                       loguru_ini=True,
                       reproducibility=False,
                       is_debug=bool(_args.is_debug))

    main(_args, _cfg)
