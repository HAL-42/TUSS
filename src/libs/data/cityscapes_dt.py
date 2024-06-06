#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/8 下午3:25
@File    : cityscape_dt.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from alchemy_cat.acplot import BGR2RGB
from alchemy_cat.data import Dataset
from alchemy_cat.py_tools import ADict
from natsort import natsorted, ns

from utils.cls_lb import lb2cls_lb

__all__ = ['kLabel2Color', 'CityScapes']

# -* 构造调色板。
kLabel2Color: np.ndarray = np.zeros((256, 3), dtype=np.uint8)
cls_colors = np.array([[128, 64, 128],
                       [244, 35, 232],
                       [70, 70, 70],
                       [102, 102, 156],
                       [190, 153, 153],
                       [153, 153, 153],
                       [250, 170, 30],
                       [220, 220, 0],
                       [107, 142, 35],
                       [152, 251, 152],
                       [70, 130, 180],
                       [220, 20, 60],
                       [255, 0, 0],
                       [0, 0, 142],
                       [0, 0, 70],
                       [0, 60, 100],
                       [0, 80, 100],
                       [0, 0, 230],
                       [119, 11, 32]], dtype=np.uint8)
kLabel2Color[:cls_colors.shape[0], :] = cls_colors


class CityScapes(Dataset):
    """
    PASCAL VOC and VOC Aug Segmentation base dataset
    """
    class_names = ['road',
                   'sidewalk',
                   'building',
                   'wall',
                   'fence',
                   'pole',
                   'traffic light',
                   'traffic sign',
                   'vegetation',
                   'terrain',
                   'sky',
                   'person',
                   'rider',
                   'car',
                   'truck',
                   'bus',
                   'train',
                   'motorcycle',
                   'bicycle']
    class_num = len(class_names)
    mean_bgr = [104.008, 116.669, 122.675]
    std_bgr = [57.375, 57.12, 58.395]
    ignore_label = 255

    def __init__(self, root: str | os.PathLike="datasets/cityscapes", split: str="train",
                 rgb_img: bool = False,
                 ret_img_file: bool=False):
        """
        Args:
            root (str): The parent dir of VOC dataset
            split (str): "train"/"val"/"trainval"
            rgb_img (bool): If True, return RGB image
            ret_img_file (bool): If True, return image file path
        """
        self.root = Path(root)
        self.split = split
        self.rgb_img = rgb_img
        self.ret_img_file = ret_img_file
        self.image_ids: list[str] = []

        self.image_dir: Path | None = None
        self.label_dir: Path | None = None

        self._set_files()

    def _set_files(self):
        assert self.split in ['train', 'val']

        self.image_dir = self.root / self.split / 'image'
        self.label_dir = self.root / self.split / 'fine_label'

        self.image_ids = natsorted([osp.splitext(img_file)[0]
                                    for img_file in os.listdir(self.image_dir) if img_file.endswith('.png')],
                                   alg=ns.PATH)

    def __len__(self):
        return len(self.image_ids)

    def get_item(self, index: int) -> ADict:
        out = ADict()

        out.img_id = img_id = self.image_ids[index]

        if self.ret_img_file:
            img = str(self.image_dir / (img_id + '.png'))
        else:
            img = cv2.imread(str(self.image_dir / (img_id + '.png')), cv2.IMREAD_COLOR)
            if self.rgb_img:
                img = BGR2RGB(img).copy()
        out.img = img

        out.lb = np.array(Image.open(self.label_dir / (img_id + '.png')), dtype=np.uint8)

        out.cls_lb = lb2cls_lb(out.lb, self.class_num, self.ignore_label)

        return out

    def get_by_img_id(self, img_id: str):
        try:
            index = self.image_ids.index(img_id)
        except ValueError:
            raise RuntimeError(f"Can't find img_id {img_id} in dataset's image_ids list")
        return self[index]

    @staticmethod
    def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
        return kLabel2Color[label_map]
