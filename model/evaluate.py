#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: yangyuxiang
Date: 2020-07-26 16:13:09
LastEditTime: 2021-04-18 21:03:59
FilePath: /Assignment2-3/model/evaluate.py
Desciption: Evaluate the loss in the dev set.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import os
import sys
import pathlib
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('..')

from dataset import collate_fn
import config


def evaluate(model, val_data, epoch):
    """Evaluate the loss for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_data (dataset.PairDataset): The evaluation data set.
        epoch (int): The epoch number.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')

    val_loss = []
    with torch.no_grad():
        DEVICE = config.DEVICE
        val_dataloader = DataLoader(dataset=val_data,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    pin_memory=True, drop_last=True,
                                    collate_fn=collate_fn)
        for batch, data in enumerate(tqdm(val_dataloader)):
            x, y, x_len, y_len, oov, len_oovs, img_vec = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
                img_vec = img_vec.to(DEVICE)
            loss = model(x,
                         y,
                         len_oovs,
                         img_vec,
                         batch=batch,
                         num_batches=len(val_dataloader))
            val_loss.append(loss.item())
    return np.mean(val_loss)
