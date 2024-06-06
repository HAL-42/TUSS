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

import numpy as np
from PIL import Image
from alchemy_cat.acplot import RGB2BGR
from alchemy_cat.data import Dataset, Subset
from alchemy_cat.py_tools import ADict
from math import ceil
from natsort import natsorted, ns

from utils.cls_lb import lb2cls_lb

__all__ = ['kStuffLabel2Color', 'COCOStuff', 'kObjLabel2Color', 'COCOObj']

# -* 构造调色板。
kStuffLabel2Color: np.ndarray = np.full((256, 3), 255, dtype=np.uint8)
cls_colors = np.array([[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                       [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                       [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                       [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                       [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                       [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                       [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                       [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                       [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                       [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                       [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                       [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                       [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                       [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                       [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                       [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                       [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                       [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                       [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                       [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                       [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                       [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                       [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                       [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                       [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                       [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                       [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                       [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                       [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                       [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                       [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                       [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                       [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                       [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                       [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                       [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                       [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                       [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                       [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                       [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                       [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                       [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                       [64, 192, 96], [64, 160, 64], [64, 64, 0]], dtype=np.uint8)
kStuffLabel2Color[:cls_colors.shape[0], :] = cls_colors

kObjLabel2Color: np.ndarray = np.full((256, 3), 255, dtype=np.uint8)
kObjLabel2Color[0, :] = 0
kObjLabel2Color[1:81, :] = kStuffLabel2Color[:80]


class COCOStuff(Dataset):

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
                   'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
                   'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
                   'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                   'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
                   'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
                   'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
                   'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
                   'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
                   'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
                   'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
                   'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
                   'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
                   'stone', 'straw', 'structural-other', 'table', 'tent',
                   'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
                   'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
                   'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                   'window-blind', 'window-other', 'wood']
    class_num = len(class_names)
    mean_bgr = None
    std_bgr = None
    ignore_label = 255

    def __init__(self, root: str | os.PathLike="datasets/cocostuff164k", split: str="train",
                 rgb_img: bool = False,
                 ret_img_file: bool=False,
                 label_suffix: str='_labelTrainIds'):
        """
        Args:
            root (str): The parent dir of VOC dataset
            split (str): "train"/"val"/"trainval"
            rgb_img (bool): If True, return RGB image
            ret_img_file (bool): If True, return image file path
            label_suffix (str): The suffix of label file
        """
        self.root = Path(root)
        self.split = split
        self.rgb_img = rgb_img
        self.ret_img_file = ret_img_file
        self.label_suffix = label_suffix

        self.image_ids: list[str] = []

        self.image_dir: Path | None = None
        self.label_dir: Path | None = None

        self._set_files()

    def _set_files(self):
        assert self.split in ['train', 'val']

        self.image_dir = self.root / 'images' / f'{self.split}2017'
        self.label_dir = self.root / 'annotations' / f'{self.split}2017'

        self.image_ids = natsorted([osp.splitext(img_file)[0]
                                    for img_file in os.listdir(self.image_dir) if img_file.endswith('.jpg')],
                                   alg=ns.PATH)

    def __len__(self):
        return len(self.image_ids)

    def get_item(self, index: int) -> ADict:
        out = ADict()

        out.img_id = img_id = self.image_ids[index]

        if self.ret_img_file:
            img = str(self.image_dir / (img_id + '.jpg'))
        else:
            pil = Image.open(self.image_dir / (img_id + '.jpg'))
            if pil.mode != 'RGB':
                pil = pil.convert('RGB')
            img = np.asarray(pil, dtype=np.uint8)
            if not self.rgb_img:
                img = RGB2BGR(img).copy()
        out.img = img

        out.lb = np.array(Image.open(self.label_dir / (img_id + f'{self.label_suffix}.png')), dtype=np.uint8)

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
        return kStuffLabel2Color[label_map]

    @classmethod
    def subset(cls,
               split_idx_num: tuple[int, int],
               root: str | os.PathLike = "datasets/cocostuff164k", split: str = "train",
               rgb_img: bool = False,
               ret_img_file: bool = False,
               label_suffix: str = '_labelTrainIds'):
        dt = cls(root=root, split=split, rgb_img=rgb_img, ret_img_file=ret_img_file, label_suffix=label_suffix)

        split_idx, split_num = split_idx_num
        assert split_idx < split_num

        step = ceil(len(dt) / split_num)
        indexes = list(range(split_idx * step, min((split_idx + 1) * step, len(dt))))

        sub_dt = Subset(dt, indexes)
        return sub_dt


class COCOObj(COCOStuff):
    class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                   'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                   'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                   'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                   'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    class_num = len(class_names)

    def __init__(self, root: str | os.PathLike="datasets/cocostuff164k", split: str="train",
                 rgb_img: bool = False,
                 ret_img_file: bool=False,
                 label_suffix: str=''):
        super().__init__(root=root, split=split, rgb_img=rgb_img, ret_img_file=ret_img_file,
                         label_suffix=label_suffix)
        self.label_dir = self.root / 'third_party' / 'masks'

    @staticmethod
    def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
        return kObjLabel2Color[label_map]

    @classmethod
    def subset(cls,
               split_idx_num: tuple[int, int],
               root: str | os.PathLike = "datasets/cocostuff164k", split: str = "train",
               rgb_img: bool = False,
               ret_img_file: bool = False,
               label_suffix: str = ''):
        return super().subset(split_idx_num=split_idx_num, root=root, split=split, rgb_img=rgb_img,
                              ret_img_file=ret_img_file, label_suffix=label_suffix)
