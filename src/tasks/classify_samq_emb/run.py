#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/12 13:26
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import pickle
import sys
import typing as t
from functools import partial
from itertools import product
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from alchemy_cat.contrib.metrics import SegmentationMetric, ClassificationMetric
from alchemy_cat.contrib.voc.scripts.colorize_voc import colorize_voc
from alchemy_cat.data import Dataset
from alchemy_cat.data.plugins import SegDataset
from alchemy_cat.py_tools import Config, ADict, CacheDir
from alchemy_cat.torch_tools import init_env
from torch.utils.data import DataLoader, dataset as torch_dataset
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.data import SAMQEmbDt
from libs.classifier.typing import SPEmbClassifier
from libs.clip.utils import OpenCLIPCreated
from utils.cls_lb import cls_lb2fg_cls
from libs.sam.custom_sam import SamAnns


class ClsEvalDt(Dataset):

    def __init__(self, emb_dir: Path | str, seg_dt: SegDataset[ADict], val: bool=False):
        self.emb_dir = Path(emb_dir)
        self.seg_dt = seg_dt
        self.val = val

        if ('val' in seg_dt.split or 'test' in seg_dt.split) and not val:
            raise ValueError("seg_dt是验证/测试集，但val为False。")

        self.emb_pkls = list(sorted(self.emb_dir.glob('*.pkl')))
        self.img_ids = [p.stem for p in self.emb_pkls]

    def __len__(self):
        return len(self.img_ids)

    def get_item(self, index: int) -> ADict:
        ret = ADict()

        emb_pkl, img_id = self.emb_pkls[index], self.img_ids[index]
        ret.img_id = img_id

        emb_data = pickle.loads(emb_pkl.read_bytes())
        ret.sp_embs = torch.from_numpy(emb_data['sp_embs'])

        ret.sps = pickle.loads(Path(emb_data['sp_file']).read_bytes())

        ori_inp = self.seg_dt.get_by_img_id(img_id)
        if not self.val:
            ret.cls_lb = ori_inp.cls_lb.astype(np.bool_)
            ret.cls_lb[0] = True  # 总是假设背景存在（弱监督没有背景有无信息）。
            ret.fg_cls = cls_lb2fg_cls(torch.from_numpy(ori_inp.cls_lb)).to(dtype=torch.long)
        else:  # 验证模式下，fg_cls全为1，即不引入任何类别信息。
            ret.cls_lb = np.ones_like(ori_inp.cls_lb, dtype=np.bool_)
            ret.fg_cls = torch.arange(self.seg_dt.class_num - 1, dtype=torch.long)
        ret.lb = ori_inp.lb

        return ret


def main(_: argparse.Namespace, cfg: str | Config,
         open_clip_created: OpenCLIPCreated=None,
         classifier: SPEmbClassifier=None) -> ADict:
    # -* 直接传入分类器。
    if classifier is not None:
        pass
    # -* 通用分类器。
    elif cfg.classifier:
        classifier = cfg.classifier.cls(**cfg.classifier.ini)
    # -* 基于CLIP的分类器。
    else:
        if open_clip_created is None:
            open_clip_created = OpenCLIPCreated.create(**cfg.clip.ini)
        clip, tokenizer = open_clip_created.model, open_clip_created.tokenizer
        classifier = cfg.txt_cls.cls(clip, tokenizer, **cfg.txt_cls.ini)

    # -* 数据集。
    seg_dt: SegDataset[ADict] = cfg.seg_dt.cls(**cfg.seg_dt.ini, ret_img_file=True)
    print(seg_dt, end="\n\n")
    cls_eval_dt = ClsEvalDt(cfg.emb.dir, seg_dt, **cfg.cls_eval_dt.ini)

    # -* 构造数据加载器。
    cls_eval_loader = DataLoader(t.cast(torch_dataset, cls_eval_dt),
                                 batch_size=None,
                                 num_workers=0,  # 实测多进程由于通信开销，速度反而变慢。
                                 collate_fn=lambda x: x,
                                 pin_memory=False,
                                 shuffle=False,
                                 drop_last=False,
                                 generator=torch.Generator().manual_seed(0),
                                 prefetch_factor=None,
                                 persistent_workers=False)

    match cfg.tune.at:
        case 'fg_bg_methods':
            rslt = tune_fg_bg_methods(cfg, cls_eval_loader, classifier)
        case {} | None:
            rslt = cls_and_eval(cfg, cls_eval_loader, classifier, cfg.seed.cal)
        case _:
            raise NotImplementedError(f"Unknown tune method: {cfg.tune}")

    rslt.cls_metric.save_metric(Path(cfg.rslt_dir) / 'eval' / 'cls', 0)
    rslt.seg_metric.save_metric(Path(cfg.rslt_dir) / 'eval' / 'seg', 0)
    if cfg.semi_ids:
        rslt.semi_cls_metric.save_metric(Path(cfg.rslt_dir) / 'eval' / 'semi_cls', 0)
        rslt.semi_seg_metric.save_metric(Path(cfg.rslt_dir) / 'eval' / 'semi_seg', 0)

    rslt.rslt_mng.save()
    colorize_voc(str(rslt.rslt_mng.seed_dir), str(Path(cfg.rslt_dir) / 'viz' / 'color_seed'),
                 num_workers=8, is_eval=False, l2c=seg_dt.label_map2color_map)

    return rslt


def tune_fg_bg_methods(cfg: str | Config, cls_eval_loader: DataLoader,
                       classifier: SPEmbClassifier) -> ADict:
    results = []
    # * 遍历所有前背景方法。
    for fg_method, bg_method in product(cfg.seed.fg_methods, cfg.seed.bg_methods):
        print(f"Current fg_method: {fg_method}, bg_method: {bg_method} ...")
        seed_cal = partial(cfg.seed.func, fg_method=fg_method, bg_method=bg_method, **cfg.seed.ini)
        results.append(rslt := cls_and_eval(cfg, cls_eval_loader, classifier, seed_cal))
        rslt.fg_method, rslt.bg_method = fg_method, bg_method

    # * 找到最优方法，非最优方法删除cache_dir。
    best_rslt = max(results, key=lambda x: x.seg_metric.mIoU)
    for rslt in [r for r in results if r is not best_rslt]:
        rslt.rslt_mng.terminate()

    # * 打印结果。
    for rslt in results:
        print(f'参数设定为：fg_method: {rslt.fg_method}, bg_method: {rslt.bg_method}，'
              f'cls_F1: {rslt.cls_metric.F1:.4f}, mIoU: {rslt.seg_metric.mIoU:.4f}')
        if cfg.semi_ids:
            print(f'--->半监督cls_F1: {rslt.semi_cls_metric.F1:.4f}, 半监督mIoU: {rslt.semi_seg_metric.mIoU:.4f}')
    print(f'最优参数设定为：fg_method: {best_rslt.fg_method}, bg_method: {best_rslt.bg_method}，'
          f'cls_F1: {best_rslt.cls_metric.F1:.4f}, mIoU: {best_rslt.seg_metric.mIoU:.4f}')
    if cfg.semi_ids:
        print(f'--->半监督cls_F1: {best_rslt.semi_cls_metric.F1:.4f}, 半监督mIoU: {best_rslt.semi_seg_metric.mIoU:.4f}')

    return best_rslt


class EmbClassifyDataManager(object):

    def __init__(self, emb_classify_rslt_dir: Path | str, cls_num: int, save_logit: bool=False):
        self.save_logit = save_logit

        # -* 配置总保存路径。
        self.emb_classify_rslt_dir = CacheDir(emb_classify_rslt_dir,
                                              '.cache_dir/classify_samq_emb/emb_classify_rslt', exist='delete')

        # -* 初始化seed保存路径。
        self.seed_dir.mkdir(parents=True, exist_ok=False)

        # -* 初始化embwise数据。
        self.cls_num = cls_num
        self.embwise_data = h5py.File(self.embwise_h5, 'w')

        self.score_dt = self.embwise_data.create_dataset('scores',
                                                         shape=(0, cls_num), maxshape=(None, cls_num),
                                                         dtype='f4', chunks=True)
        self.cls_lb_dt = self.embwise_data.create_dataset('cls_lb',
                                                          shape=(0, cls_num), maxshape=(None, cls_num),
                                                          dtype='?', chunks=True)
        self.logit_dt: h5py.Dataset | None = None  # 运行时动态初始化。

    @property
    def seed_dir(self) -> Path:
        return Path(self.emb_classify_rslt_dir) / 'seed'

    @property
    def embwise_h5(self) -> Path:
        return Path(self.emb_classify_rslt_dir) / 'embwise.h5'

    def write_sample(self, img_id: str, seed: torch.Tensor, seeded_sps: SamAnns, cls_lb: np.ndarray,
                     logit: torch.Tensor=None):
        # -* 参数检查、处理。
        if self.save_logit:
            assert logit is not None, "logit is None, but save_logit is True."
            logit = logit.cpu().numpy()

        # -* 保存seed。
        cv2.imwrite(str(self.seed_dir / f'{img_id}.png'), seed.to(dtype=torch.uint8).cpu().numpy())

        # -* 保存embwise数据。
        score = torch.stack(seeded_sps.list_key('score'), dim=0).cpu().numpy()  # (S, cls_num)
        sp_num = score.shape[0]

        self.score_dt.resize((self.score_dt.shape[0] + sp_num, self.cls_num))
        self.score_dt[-sp_num:, cls_lb] = score

        self.cls_lb_dt.resize((self.cls_lb_dt.shape[0] + sp_num, self.cls_num))
        self.cls_lb_dt[-sp_num:, :] = cls_lb[None, :]

        if self.save_logit:
            if self.logit_dt is None:
                self.logit_dt = self.embwise_data.create_dataset('logits',
                                                                 shape=(0, logit.shape[1]),
                                                                 maxshape=(None, logit.shape[1]),
                                                                 dtype='f4', chunks=True)
            self.logit_dt.resize((self.logit_dt.shape[0] + sp_num, logit.shape[1]))
            self.logit_dt[-sp_num:, :] = logit

    def _close_h5(self):
        self.score_dt = self.cls_lb_dt = self.logit_dt = None
        self.embwise_data.close()

    def terminate(self):
        self._close_h5()
        self.emb_classify_rslt_dir.terminate()

    def save(self):
        self._close_h5()
        self.emb_classify_rslt_dir.save()

    def __del__(self):
        self.terminate()


def cls_and_eval(cfg: str | Config, cls_eval_loader: DataLoader,
                 classifier: SPEmbClassifier, seed_cal: t.Callable) -> ADict:
    rslt = ADict()
    seg_dt = t.cast(ClsEvalDt, cls_eval_loader.dataset).seg_dt

    # -* 配置路径。
    rslt.rslt_mng = rslt_mng = EmbClassifyDataManager(Path(cfg.rslt_dir) / 'emb_classify_rslt',
                                                      cls_num=seg_dt.class_num,
                                                      save_logit=bool(cfg.seed.save_logit))

    # -* 配置评价器。
    rslt.seg_metric = seg_metric = SegmentationMetric(seg_dt.class_num, seg_dt.class_names)
    rslt.cls_metric = cls_metric = ClassificationMetric(seg_dt.class_num, seg_dt.class_names)
    if cfg.semi_ids:  # 若指定了半监督id，则配置半监督评价器（评价无标签样本的性能）。
        rslt.semi_seg_metric = semi_seg_metric = SegmentationMetric(seg_dt.class_num, seg_dt.class_names)
        rslt.semi_cls_metric = semi_cls_metric = ClassificationMetric(seg_dt.class_num, seg_dt.class_names)
        semi_ids = SAMQEmbDt.read_semi_ids(cfg.semi_ids)

    # -* 遍历所有图片。
    for inp in tqdm(cls_eval_loader, dynamic_ncols=True, desc='分类', unit='张', miniters=10):
        img_id, sp_embs, sps, fg_cls, lb, cls_lb = inp.img_id, inp.sp_embs, inp.sps, inp.fg_cls, inp.lb, inp.cls_lb
        sp_embs, fg_cls = sp_embs.to('cuda', non_blocking=True), fg_cls.to('cuda', non_blocking=True)

        # -** 跳过已有。（由于用了CacheDir，其实不会跳过）
        if (seed_png := rslt_mng.seed_dir / f'{img_id}.png').is_file():
            print(f"[跳过] {seed_png}已存在，跳过。")
            continue

        # -* 分类。
        sp_cls_logit = classifier(sp_embs, img_id)  # (M,K)

        # -* 获取预测Mask。
        seed, seeded_sps = seed_cal(sps, sp_cls_logit, fg_cls)  # (H,W)
        seed: torch.Tensor
        seeded_sps: SamAnns

        # -* 更新评价器。
        seg_metric.update(seed_ := seed.cpu().numpy(), lb)
        cls_metric.update(cls_seed_ := np.array(seeded_sps.list_key('seed')),
                          vote_lb_ := np.array(seeded_sps.list_key('vote_lb')))
        if cfg.semi_ids and img_id not in semi_ids:  # 若指定了半监督/有标签id，且当前不是有标签id，则更新半监督评价器。
            semi_seg_metric.update(seed_, lb)
            semi_cls_metric.update(cls_seed_, vote_lb_)

        # -* 保存seed和score.
        rslt_mng.write_sample(img_id, seed, seeded_sps, cls_lb, sp_cls_logit)

    print(f"cls_ACC: {cls_metric.ACC:.4f}, cls_F1: {cls_metric.F1:.4f}, mIoU: {seg_metric.mIoU:.4f}")

    return rslt


if __name__ == '__main__':
    # -* 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    _args = parser.parse_args()

    # -* 初始化环境。
    _, _cfg = init_env(is_cuda=True,
                       is_benchmark=False,
                       is_train=False,
                       config_path=_args.config,
                       experiments_root="experiment",
                       rand_seed=0,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=bool(_args.is_debug))

    main(_args, _cfg)
