#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/4/14 下午11:14
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import sklearn.preprocessing as prep
from alchemy_cat.py_tools import Config
from alchemy_cat.torch_tools import init_env
from loguru import logger
from sklearn.metrics import silhouette_score

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.emb_analysis.pca import PCARet
from libs.data import SAMQEmbDt


@dataclass
class RemovePCPacket(object):
    # -* 输入量。
    sp_emb: np.ndarray = None  # [N,d],f4
    Y_bar: np.ndarray = None  # [N],i8
    Y: np.ndarray = None  # [N],i8
    num_cls: int = None
    sp_emb_normed: np.ndarray = None  # [N,d],f4

    # -* 中间量。
    weight: np.ndarray = None  # [N],f4
    UUT: np.ndarray = None  # [d,d],f4

    # -* 输出量。
    _refined_sp_emb: np.ndarray = None  # [N,K],f4
    _refined_sp_emb_normed: np.ndarray = None  # [N,K],f4

    @classmethod
    def from_gather_and_classify_result(cls,
                                        emb_dir: str | os.PathLike,
                                        emb_classify_rslt_dir: str | os.PathLike=None) -> 'RemovePCPacket':
        logger.info(f"从 {emb_dir} 获取sp_emb和Y...")
        emb_dt = SAMQEmbDt(emb_dir)
        sp_emb = np.array(emb_dt.emb_data['sp_embs'], copy=False)
        Y = np.argmax(probs := np.array(emb_dt.emb_data['probs'], copy=False), axis=1)
        num_cls = probs.shape[1]

        if emb_classify_rslt_dir is not None:
            logger.info(f"从 {emb_classify_rslt_dir} 获取Y_bar...")
            with h5py.File(Path(emb_classify_rslt_dir) / 'embwise.h5', 'r') as f:
                score = f['scores'][...]
                assert score.shape == probs.shape
                Y_bar = np.argmax(score, axis=1)
        else:
            logger.info(f"未提供emb_classify_rslt_dir，使用Y作为Y_bar...")
            Y_bar = Y

        sp_emb_normed = prep.normalize(sp_emb, norm='l2')  # [N,d],f4

        return cls(sp_emb=sp_emb, Y_bar=Y_bar, Y=Y, num_cls=num_cls, sp_emb_normed=sp_emb_normed).cal_weight()

    def cal_weight(self) -> 'RemovePCPacket':
        """计算每个类别的权重，并作用回每个样本"""
        total_num = self.Y_bar.shape[0]
        cls_num = np.bincount(self.Y_bar, minlength=self.num_cls)
        cls_weight = total_num / (cls_num * self.num_cls + 1)
        self.weight = cls_weight[self.Y_bar]

        return self

    @property
    def refined_sp_emb(self) -> np.ndarray:
        return self._refined_sp_emb

    @refined_sp_emb.setter
    def refined_sp_emb(self, refined_sp_emb: np.ndarray):
        self._refined_sp_emb = refined_sp_emb
        self._refined_sp_emb_normed = prep.normalize(refined_sp_emb, norm='l2')

    @property
    def refined_sp_emb_normed(self) -> np.ndarray:
        return self._refined_sp_emb_normed


@dataclass
class RemovePCRet(object):
    ini_SC: float = None
    refined_SC: float = None
    UUT: np.ndarray = None  # [d,d],f4

    def save(self, rslt_dir: Path):
        # -* 记录轮廓系数。
        os.makedirs(rslt_dir / 'eval', exist_ok=True)
        (rslt_dir / 'eval' / 'statistics.json').write_text(json.dumps({'ini_SC': float(self.ini_SC),
                                                                       'refined_SC': float(self.refined_SC)}))

        # -* 保存自身。
        (rslt_dir / 'remove_pc_ret.pkl').write_bytes(pickle.dumps(self))


def main(_: argparse.Namespace, cfg: str | Config) -> RemovePCRet:
    rslt = RemovePCRet()

    # -* 构造出主成分去除数据包。
    logger.info("构造出成分去除数据包...")
    packet = RemovePCPacket.from_gather_and_classify_result(**cfg.remove_pc_packet.ini)

    # -* 计算初始轮廓系数。
    # logger.info("计算初始轮廓系数...")
    # rslt.ini_SC = silhouette_score(packet.sp_emb_normed, packet.Y)
    # logger.info(f"初始轮廓系数：{rslt.ini_SC}")
    rslt.ini_SC = -233.  # -* 由于计算轮廓系数的时间过长，故直接设置为-233.。

    # -* 获取主成分。
    pca_ret: PCARet = cfg.pca.cls(**cfg.pca.ini)(packet)
    C = pca_ret.components  # [d,d],f4

    var = pca_ret.explained_variance  # [d],f4
    removed_var, total_var = var[:cfg.uut.remove_num].sum(), var.sum()
    logger.info(f"去除{cfg.uut.remove_num}个主成分；"
                f"去除的方差：{removed_var:.2f}，剩余方差：{total_var - removed_var:.2f}；"
                f"去除的方差占比：{removed_var / total_var:.2f}。")

    # -* 计算UUT
    U = C.T[:, cfg.uut.remove_num:]
    rslt.UUT = packet.UUT = U @ U.T

    # -* 计算去除主成分后的sp_emb
    packet.refined_sp_emb = packet.sp_emb @ packet.UUT

    # -* 计算去除主成分后的轮廓系数。
    logger.info("计算去除主成分后的轮廓系数...")
    rslt.refined_SC = silhouette_score(packet.refined_sp_emb_normed, packet.Y, metric='cosine')
    logger.info(f"去除主成分前、后的轮廓系数：{rslt.ini_SC} -> {rslt.refined_SC}")

    # -* 保存结果。
    logger.info("保存结果...")
    rslt.save(Path(cfg.rslt_dir))

    return rslt


if __name__ == '__main__':
    # -* 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    _args = parser.parse_args()

    # -* 初始化环境。
    _, _cfg = init_env(is_cuda=False,
                       is_benchmark=False,
                       is_train=False,
                       config_path=_args.config,
                       rand_seed=0,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       loguru_ini=True,
                       reproducibility=False,
                       is_debug=bool(_args.is_debug))

    main(_args, _cfg)
