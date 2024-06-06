#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/2/20 23:09
@File    : argparser_tools.py
@Software: PyCharm
@Desc    : 
"""
import argparse
from itertools import chain

__all__ = ['recover_cli_from_args']


def recover_cli_from_args(args: argparse.Namespace, shell: bool=False) -> str | list[str]:
    """从argparse.Namespace中恢复cli命令。

    Args:
        args: argparse.Namespace。
        shell: 是否是shell命令。

    Returns:
        str: cli命令。
    """
    cli = list(chain.from_iterable([(f'--{k}', str(v)) for k, v in vars(args).items()]))
    if shell:
        cli = ' '.join(cli)
    return cli
