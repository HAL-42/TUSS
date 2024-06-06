#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/5/12 23:19
@File    : sp字段替换.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=Path, required=True)
parser.add_argument('--target_dir', type=Path, required=True)
parser.add_argument('--new_sp_root', type=Path, required=True)
args = parser.parse_args()

src_emb_dir: Path = args.src_dir / 'emb'
target_emb_dir: Path = args.target_dir / 'emb'
target_emb_dir.mkdir(parents=True, exist_ok=True)

for emb_pkl in tqdm(list(sorted((f for f in src_emb_dir.iterdir() if f.suffix == '.pkl'))),
                    desc='Processing emb pkl files', unit='file', dynamic_ncols=True):
    data = pickle.loads(emb_pkl.read_bytes())
    
    ori_sp_file = Path(data['sp_file'])
    new_sp_file = args.new_sp_root / ori_sp_file.name
    assert new_sp_file.is_file(), f'{new_sp_file} is not a file'

    data['sp_file'] = str(new_sp_file)

    (target_emb_dir / emb_pkl.name).write_bytes(pickle.dumps(data))
