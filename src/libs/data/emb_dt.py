#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/9 1:26
@File    : emb_dt.py
@Software: PyCharm
@Desc    : embedding数据集。读取每个图片的emb，（拼接为h5），并支持index查询获取。
"""
import bisect
import itertools
import json
import os
import pickle
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from alchemy_cat.data import Dataset
from alchemy_cat.py_tools import ADict, hash_str
from tqdm import tqdm

from libs.sam import SamAnns
from libs.seeding.confidence import complement_entropy

__all__ = ['SAMQEmbDt']


class SAMQEmbDt(Dataset):
    def __init__(self, root: str | os.PathLike,
                 semi_ids: str | os.PathLike | list[str]=None, conf_method: dict[str, ...]=None):
        super().__init__()
        self.root = Path(root)
        self.conf_method = conf_method

        # -*检查是否已经收集了sample pkls。
        if not self.is_emb_pkls_collected:
            print('Collecting sample pkls...')
            self.collect_and_store_emb_pkls()
        # -*读取emb数据。
        # -** 如果emb_data_file大于5个G，不载入内存：
        # if self.emb_data_file.stat().st_size > 5 * 1024 ** 3:
        #     self.emb_data = h5py.File(self.emb_data_file, 'r')
        # else:
        with h5py.File(self.emb_data_file, 'r') as f:
            self.emb_data = {k: np.array(v) for k, v in f.items()}

        # -*读取emb元数据。
        with open(self.emb_meta_file, 'rb') as f:
            self.emb_meta = pickle.load(f)
        # -* 记录伪标签概率。
        self.psudo_prob: h5py.File | dict[str, np.ndarray] | None = None

        # -* semi相关数据。
        self.semi_ids: list[str] | None = None
        self.semi_mask: np.ndarray | None = None
        self._parse_semi_ids(semi_ids)

        # -* 构造probs和conf。
        self.probs: np.ndarray | None = None
        self.cls_lb: np.ndarray | None = None
        self.soft_lb: np.ndarray | None = None
        self.conf: np.ndarray | None = None
        self._construct_probs()

    def _set_psudo_prob_h5(self, psudo_prob_h5: str | os.PathLike=None):
        psudo_prob_h5 = Path(psudo_prob_h5) if psudo_prob_h5 else None
        # -** 若psudo_prob_h5不存在，则不载入。
        if not psudo_prob_h5:
            self.psudo_prob = None
        # -** 若psudo_prob_h5存在且大于5G，不载入内存。
        elif psudo_prob_h5.stat().st_size > 5 * 1024 ** 3:
            self.psudo_prob = h5py.File(psudo_prob_h5, 'r')
        # -** 若psudo_prob_h5存在且小于5G，载入内存。
        else:
            with h5py.File(psudo_prob_h5, 'r') as f:
                self.psudo_prob = {k: np.array(v) for k, v in f.items()}

    @classmethod
    def from_psudo_mask(cls, root: str | os.PathLike, psudo_mask_dir: str | os.PathLike | None,
                        ignore_label: int=255, semi_ids: str | os.PathLike | list[str]=None) -> 'SAMQEmbDt':
        warnings.warn("该方法不再维护。", DeprecationWarning)

        dt = cls(root, semi_ids=semi_ids)

        # -* 若psudo_mask_dir为None，则不需要处理。
        if psudo_mask_dir is None:
            return dt

        # -* 配置路径。
        psudo_mask_dir = Path(psudo_mask_dir)
        psudo_mask_dir2psudo_prob_h5_json = dt.emb_data_dir / 'psudo_mask_dir2psudo_prob_h5.json'

        # -* 读取psudo_mask_dir2psudo_prob_h5.json。
        if psudo_mask_dir2psudo_prob_h5_json.is_file():
            psudo_mask_dir2psudo_prob_h5: dict[str, str] = json.loads(psudo_mask_dir2psudo_prob_h5_json.read_text())
        else:
            psudo_mask_dir2psudo_prob_h5: dict[str, str] = {}

        # -* 若psudo_mask_dir已经转换为psudo_prob_h5，则直接载入。
        if str(psudo_mask_dir) in psudo_mask_dir2psudo_prob_h5:
            dt._set_psudo_prob_h5(psudo_mask_dir2psudo_prob_h5[str(psudo_mask_dir)])
            return dt

        # -* 若psudo_mask_dir未转换为psudo_prob_h5，则转换并保存。
        # -* 处理路径，并保存到psudo_mask_dir2psudo_prob_h5.json。
        psudo_prob_h5 = dt.emb_data_dir / 'psudo_prob' / f'{hash_str(str(psudo_mask_dir))}.h5'
        psudo_prob_h5.parent.mkdir(parents=True, exist_ok=True)
        psudo_mask_dir2psudo_prob_h5[str(psudo_mask_dir)] = str(psudo_prob_h5)

        # -* 初始化psudo_prob_h5。
        with h5py.File(psudo_prob_h5, 'w') as data_h5:
            probs_dt = data_h5.create_dataset('probs',
                                              shape=(0, dt.cls_num), maxshape=(None, dt.cls_num),
                                              dtype='f4', chunks=True)

            # -* 遍历所有emb_pkl。
            for emb_pkl in tqdm(dt.emb_pkls, desc='Collecting Psudo', unit='samples', dynamic_ncols=True):
                # -* 读取伪标签。
                psudo_mask = np.asarray(Image.open(psudo_mask_dir / f'{emb_pkl.stem}.png'), dtype=np.uint8)
                psudo_mask = torch.from_numpy(psudo_mask).to(device='cuda', non_blocking=True).to(dtype=torch.long)

                # -*读取 pkl 文件，获取每个样本的所有超像素。
                _, _, _, sps, _ = dt.read_emb_pkl(emb_pkl)
                sps: SamAnns
                sps.stack_data(masks=True, areas=True, device='cuda', replace=False)
                sp_num = len(sps)

                # -* 获取该超像素基于投票法的分类概率。
                sp_lb_vote_prob = torch.zeros((sp_num, dt.cls_num), device='cuda', dtype=torch.float32)
                for i, m in enumerate(sps.data.masks):  # TODO：将lb转为one-hot，这个for可以被优化为批量操作。
                    lb_on_mask_ = psudo_mask[m]
                    lb_on_mask = lb_on_mask_[lb_on_mask_ != ignore_label]
                    cls_in_sp, cls_count_in_sp = torch.unique(lb_on_mask, return_counts=True)

                    # TODO 考虑mask的置信度。
                    sp_lb_vote_prob[i, cls_in_sp] = cls_count_in_sp.float() / cls_count_in_sp.sum(dtype=torch.float32)

                # -* 将伪标签概率追加到 HDF5 文件中。
                probs_dt.resize((probs_dt.shape[0] + sp_num, probs_dt.shape[1]))
                probs_dt[-sp_num:, :] = sp_lb_vote_prob.cpu().numpy()

        # -* 保存更新后的psudo_mask_dir2psudo_prob_h5.json。
        psudo_mask_dir2psudo_prob_h5_json.write_text(json.dumps(psudo_mask_dir2psudo_prob_h5,
                                                                indent=4, ensure_ascii=False))

        # -* 载入psudo_prob_h5。
        dt._set_psudo_prob_h5(psudo_prob_h5)

        # -* 重建probs。
        dt._construct_probs()

        return dt

    @classmethod
    def from_embwise_h5(cls, root: str | os.PathLike,
                        embwise_h5: os.PathLike | str | None, prob2soft_lb: dict[str, ...] | str=None,
                        score2prob: dict[str, ...] | str=None,
                        prob_dt_name: str='scores', semi_ids: str | os.PathLike | list[str]=None,
                        conf_method: dict[str, ...]=None) -> 'SAMQEmbDt':
        # -* 处理prob2soft_lb和score2prob。
        assert (prob2soft_lb, score2prob).count(None) == 1, "prob2soft_lb和score2prob必须有且仅有一个为None。"
        if prob2soft_lb is None:
            prob2soft_lb = score2prob
            warnings.warn("score2prob 参数已经废弃, 请使用 prob2soft_lb 参数。", DeprecationWarning)

        # -* 初始化dt。
        dt = cls(root, semi_ids=semi_ids, conf_method=conf_method)

        # -* 若没有伪真值，则不需要处理。
        if embwise_h5 is None:
            return dt

        # -* 读取伪真值，计算软标签。
        with h5py.File(embwise_h5, 'r') as f:
            prob: np.ndarray = f[prob_dt_name][...]
            cls_lb: np.ndarray = f['cls_lb'][...]

        dt.psudo_prob = {'probs': prob, 'cls_lb': cls_lb}
        if prob2soft_lb == 'identical':
            dt.psudo_prob['soft_lb'] = prob
        else:
            prob: torch.Tensor = torch.from_numpy(prob).to('cuda', non_blocking=True)
            cls_lb: torch.Tensor = torch.from_numpy(cls_lb).to('cuda', non_blocking=True)
            match prob2soft_lb:
                case {'norm': 'softmax', 'gamma': float(gamma)}:
                    prob *= gamma
                    prob[~cls_lb] = float('-inf')
                    soft_lb = F.softmax(prob, dim=1)
                case {'norm': 'L1'}:
                    soft_lb = F.normalize(prob, p=1, dim=1)
                case _:
                    raise ValueError(f'Unknown prob2soft_lb: {prob2soft_lb}')
            dt.psudo_prob['soft_lb'] = soft_lb.cpu().numpy()

        # -* 重建probs。
        dt._construct_probs()

        return dt

    def __del__(self):
        if hasattr(self, 'emb_data') and isinstance(self.emb_data, h5py.File):
            self.emb_data.close()
        if hasattr(self, 'psudo_prob') and isinstance(self.psudo_prob, h5py.File):
            self.psudo_prob.close()

    def __len__(self) -> int:
        return self.emb_meta['emb_num']

    @property
    def labeled_unlabeled_indices(self):
        assert self.semi_mask is not None
        return np.nonzero(self.semi_mask)[0], np.nonzero(~self.semi_mask)[0]

    @staticmethod
    def read_semi_ids(semi_ids_file: str | os.PathLike, img_ids: list[str]=None) -> list[str]:
        semi_ids = (Path(semi_ids_file) / 'labeled.txt').read_text().splitlines()

        if img_ids is not None:  # -* 若img_ids不为None，则做安全检查。
            img_ids_set = set(img_ids)
            semi_ids_set = set(semi_ids)

            if (unlabeled_txt := Path(semi_ids_file) / 'unlabeled.txt').is_file():
                unlabeled_ids_set = set(unlabeled_txt.read_text().splitlines())
                assert unlabeled_ids_set.isdisjoint(set(semi_ids))  # 有无标签的id不能重复。
                assert (unlabeled_ids_set | semi_ids_set) == img_ids_set  # 有无标签的id合并应该等于img_ids。

            assert len(semi_ids_set) == len(semi_ids)  # 检查是否有重复的semi_ids。
            assert semi_ids_set.issubset(img_ids_set)  # 检查是否有img_ids中不存在的semi_ids。

        return semi_ids

    def _parse_semi_ids(self, semi_ids: str | os.PathLike | list[str]):
        # -* 若为None，则不是semi数据集。
        if semi_ids is None:
            self.semi_ids, self.semi_mask = None, None
            return

        # -* 若为list[str]，则载入semi_ids。
        if isinstance(semi_ids, list):
            self.semi_ids = semi_ids
        else:
            self.semi_ids = self.read_semi_ids(semi_ids, self.emb_meta['img_ids'])

        # -* 记录semi_mask。
        self.semi_mask = np.zeros(self.emb_meta['emb_num'], dtype=np.bool_)
        semi_indices = np.asarray(list(itertools.chain.from_iterable((range(*self.img_id_start_end_indices[semi_id])
                                                                      for semi_id in self.semi_ids))), dtype=np.int64)
        self.semi_mask[semi_indices] = True

    def _construct_probs(self):
        # -* 若是一个semi数据集，则有标签部分设置为prob，无标签部分若有psudo，则设置为psudo，否则设置为0。
        if self.semi_ids is not None:
            if self.psudo_prob is not None:
                probs = self.psudo_prob['probs'].copy()
                cls_lb = self.psudo_prob['cls_lb'].copy()
                soft_lb = self.psudo_prob['soft_lb'].copy()
            else:
                probs = np.zeros_like(self.emb_data['probs'])
                cls_lb = np.ones_like(probs, dtype=bool)
                soft_lb = np.zeros_like(self.emb_data['probs'])
            probs[self.semi_mask, :] = self.emb_data['probs'][self.semi_mask, :]
            cls_lb[self.semi_mask, :] = self.emb_data['probs'][self.semi_mask, :] > 1e-6
            soft_lb[self.semi_mask, :] = self.emb_data['probs'][self.semi_mask, :]
        # -* 若不是semi数据集，且有psudo，则设置为psudo，否则设置为prob。
        elif self.psudo_prob is not None:
            probs = self.psudo_prob['probs']
            cls_lb = self.psudo_prob['cls_lb']
            soft_lb = self.psudo_prob['soft_lb']
        else:
            soft_lb = probs = self.emb_data['probs']
            cls_lb = probs > 1e-6

        self.probs, self.cls_lb, self.soft_lb = probs, cls_lb, soft_lb
        self._construct_conf()

    def _construct_conf(self):
        def entropy_conf(probs: np.ndarray) -> np.ndarray:
            c = complement_entropy(probs)
            c[np.isnan(c)] = 0  # 有些样本的概率全为0，导致entropy为nan。
            c /= c.max()
            return c

        def cls_weight_conf(probs: np.ndarray, cls_num: int) -> np.ndarray:
            lb = np.argmax(probs, axis=1)
            cls_count = np.bincount(lb, minlength=cls_num)
            cls_weight = (lb.shape[0] / cls_num) / cls_count
            return cls_weight[lb]

        match self.conf_method:
            case None | {'method': 'no_conf'}:
                conf = None
            case {'method': 'entropy'}:
                conf = entropy_conf(self.probs)
            case {'method': 'cls_weight'}:
                conf = cls_weight_conf(self.probs, self.cls_num)
            case {'method': 'cls_weight_entropy'}:
                conf = cls_weight_conf(self.probs, self.cls_num) * entropy_conf(self.probs)
            case _:
                raise ValueError(f'Unknown conf_method: {self.conf_method}')
        self.conf = conf

    # -* 数据读取相关。

    @property
    def emb_dim(self) -> int:
        return self.emb_meta['emb_dim']

    @property
    def cls_num(self) -> int:
        return self.emb_meta['cls_num']

    @property
    def img_id_start_end_indices(self) -> dict[str, tuple[int, int]]:
        start_end_indices = zip(self.emb_meta['img_id_start_indexes'],
                                self.emb_meta['img_id_start_indexes'][1:] + [self.emb_meta['emb_num']], strict=True)

        return {img_id: (start, end)
                for img_id, (start, end) in zip(self.emb_meta['img_ids'], start_end_indices)}

    def get_item(self, index: int) -> ADict:
        out = ADict()

        # -*读取emb数据。
        out.emb = self.emb_data['sp_embs'][index, :]
        out.area = self.emb_data['areas'][index]
        out.soft_lb = self.soft_lb[index, :]
        if self.conf is not None:
            out.conf = self.conf[index]

        # -*找到对应的图片id。
        emb_from_nth_sample = bisect.bisect_right(self.emb_meta['img_id_start_indexes'], index) - 1
        out.img_id = self.emb_meta['img_ids'][emb_from_nth_sample]
        out.nth_emb_in_sample = index - self.emb_meta['img_id_start_indexes'][emb_from_nth_sample]

        return out

    # -* 存储结构相关。

    @property
    def emb_dir(self) -> Path:
        return self.root

    @property
    def emb_pkls(self) -> list[Path]:
        return list(sorted((f for f in self.emb_dir.iterdir() if f.suffix == '.pkl')))

    @property
    def emb_data_dir(self) -> Path:
        return self.root.parent / 'emb_collected'

    @property
    def emb_data_file(self) -> Path:
        return self.emb_data_dir / 'emb.h5'

    @property
    def emb_meta_file(self) -> Path:
        return self.emb_data_dir / 'emb.pkl'

    @property
    def is_emb_pkls_collected(self) -> bool:
        return self.emb_data_file.is_file() and self.emb_meta_file.is_file()

    @staticmethod
    def read_emb_pkl(emb_pkl: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, SamAnns, str]:
        with open(emb_pkl, 'rb') as f:
            data = pickle.load(f)
            sp_embs: np.ndarray = data['sp_embs']
            sp_file: str = data['sp_file']

        # 读取 sp_file 中的内容
        with open(sp_file, 'rb') as f:
            anns: SamAnns = pickle.load(f)
            areas: np.ndarray = np.array([ann['area'] for ann in anns], dtype=np.int64)
            probs: np.ndarray = np.stack([ann['sp_lb_vote_prob'] for ann in anns], axis=0)

        assert sp_embs.shape[0] == areas.shape[0] == probs.shape[0]

        return sp_embs, areas, probs, anns, sp_file

    def collect_and_store_emb_pkls(self):
        # -*建立文件夹。
        self.emb_data_dir.mkdir(parents=True, exist_ok=True)

        # -*初始化 HDF5 文件和元数据。
        meta = {'img_ids': [], 'img_id_start_indexes': [], 'sp_files': [],
                'emb_num': None,
                'emb_dim': None,
                'cls_num': None}

        # -*读取第一个 pkl 文件，以确定 HDF5 文件的形状。
        sp_embs, areas, probs, _, _ = self.read_emb_pkl((emb_pkls := self.emb_pkls)[0])
        meta['emb_dim'] = sp_embs.shape[1]
        meta['cls_num'] = probs.shape[1]

        with h5py.File(self.emb_data_file, 'w') as data_h5:
            # -*准备在 HDF5 文件中创建数据集
            sp_embs_dt = data_h5.create_dataset('sp_embs',
                                                shape=(0, sp_embs.shape[1]), maxshape=(None, sp_embs.shape[1]),
                                                dtype='f4', chunks=True)
            areas_dt = data_h5.create_dataset('areas',
                                              shape=(0,), maxshape=(None,),
                                              dtype='i4', chunks=True)
            probs_dt = data_h5.create_dataset('probs',
                                              shape=(0, probs.shape[1]), maxshape=(None, probs.shape[1]),
                                              dtype='f4', chunks=True)

            # -*遍历根目录下的所有 pkl 文件
            current_index = 0
            for emb_pkl in tqdm(emb_pkls, desc='Collecting', unit='samples', dynamic_ncols=True):
                # -*读取 pkl 文件。
                sp_embs, areas, probs, _, sp_file = self.read_emb_pkl(emb_pkl)
                sp_num = sp_embs.shape[0]

                # -*将数据追加到 HDF5 文件中
                sp_embs_dt.resize((sp_embs_dt.shape[0] + sp_num, sp_embs_dt.shape[1]))
                sp_embs_dt[-sp_num:, :] = sp_embs

                areas_dt.resize((areas_dt.shape[0] + sp_num,))
                areas_dt[-sp_num:] = areas

                probs_dt.resize((probs_dt.shape[0] + sp_num, probs_dt.shape[1]))
                probs_dt[-sp_num:, :] = probs

                # -*记录当前 id 的索引范围
                meta['img_ids'].append(emb_pkl.stem)
                meta['img_id_start_indexes'].append(current_index)
                current_index += sp_num
                meta['sp_files'].append(sp_file)

        meta['emb_num'] = current_index

        with open(self.emb_meta_file, 'wb') as f:
            pickle.dump(meta, f)
