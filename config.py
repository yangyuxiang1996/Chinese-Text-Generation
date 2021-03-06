#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-07-26 16:13:08
LastEditTime: 2020-08-06 21:09:10
FilePath: /JD_project_2/model/config.py
Desciption: Define configuration parameters.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import os
from typing import Optional

import torch

# General
root_path = os.path.abspath(os.path.dirname(__file__))
is_cuda = True
device = torch.device('cuda') if is_cuda else torch.device('cpu')

# Data
max_vocab_size = 20000
embed_file: Optional[str] = None  # use pre-trained embeddings
train_data_path: str = os.path.join(root_path, 'files/train.txt')
val_data_path: Optional[str] = os.path.join(root_path, 'files/dev.txt')
test_data_path: Optional[str] = os.path.join(root_path, 'files/test.txt')
stop_word_file = os.path.join(root_path, 'files/HIT_stop_words.txt')
max_src_len: int = 200  # exclusive of special tokens such as EOS
max_tgt_len: int = 80  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 80
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0
data_folder = '/home/user10000761/dataset/京东智能营销文本/8a5a71a4-4d1f-4d67-b1f3-ea893e99488d/file'
img_vecs = os.path.join(root_path, 'files/img_vecs.txt')
sample_path = os.path.join(root_path, 'files/samples.txt')
vocab = os.path.join(root_path, 'files/vocab.pkl')
# Training
hidden_size: int = 512
dec_hidden_size: Optional[int] = 512
embed_size: int = 256
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.15
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 10
batch_size = 16
pointer = True
coverage = True
fine_tune = True
img_feat = True
max_grad_norm = 5.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1
patience = 5
img_vec_dim = 1000

model_name = 'baseline'
if pointer:
    model_name = 'pgn'
    if img_feat:
        model_name += '_multimodal'
    if coverage:
        model_name += '_cov'
model_name += '_%s' % str(batch_size) + '_%s' % str(learning_rate)
if fine_tune:
    model_for_ft = model_name.replace('_cov', '')
    model_name += '_ft'
        

encoder_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/encoder.pt')
decoder_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/decoder.pt')
attention_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/attention.pt')
reduce_state_save_name = os.path.join(root_path, 'saved_model/' + model_name + '/reduce_state.pt')
losses_path = os.path.join(root_path, 'saved_model/' + model_name + '/val_losses.txt')
log_path = os.path.join(root_path, 'runs/' + model_name)


# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 0.6

#Inference and evaluation
rouge_path = os.path.join(root_path, 'files/rouge_result.txt')