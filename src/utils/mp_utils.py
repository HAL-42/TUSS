#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/12/19 23:29
@File    : run_proc.py
@Software: PyCharm
@Desc    : 
"""
import torch.multiprocessing as mp

__all__ = ['run_func_as_proc']


def run_func_as_proc(func, *args):
    """Run a function as a process"""
    with mp.Pool(processes=1) as pool:
        res = pool.starmap(func, (args,))[0]

    return res
