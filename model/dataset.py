#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: yangyuxiang
Date: 2020-07-26 16:13:09
LastEditTime: 2021-05-09 20:26:56
FilePath: /Assignment2-3/model/dataset.py
Desciption: Define the format of data used in the model.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import sys
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset

sys.path.append('..')
from utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from vocab import Vocab
import config
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S'
                    )


class SamplesDataset(Dataset):
    """The class represents a sample.

    """
    def __init__(self,
                 filename,
                 tokenizer: Callable = simple_tokenizer,
                 vocab=None
                 ):
        logging.info("Reading dataset %s...\n" % filename)
        self.filename = filename
        self.tokenizer = tokenizer
        self.img_vecs = self.get_img_vecs(config.img_vecs)
        self.samples = self.build_samples(filename)
        self.vocab = vocab if vocab else self.build_vocab(config.embed_file)
        self._len = len(self.samples)

    def build_samples(self, filename):
        """Build the samples for the data set.

        Args:
            filename (str):
            The file path of the data.

        Returns:
            samples (list(dict)): All samples each represented as a dictionary.
        """
        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)

            samples = []
            count = 0
            for i, line in enumerate(f):
                count += 1
                if count % 1000 == 0:
                    logging.info(f'processing sample no.{count}.')
                sample = {}
                try:
                    source, target, cate, imgid = line.strip().split('\t')
                except Exception:
                    logging.info('exception happened when processing: ' + line)
                src = self.tokenizer(source)
                if config.max_src_len and len(src) > config.max_src_len:
                    if config.truncate_src:
                        src = src[:config.max_src_len]
                    else:
                        continue
                sample['src'] = src

                tgt = self.tokenizer(target)
                if config.max_tgt_len and len(tgt) > config.max_tgt_len:
                    if config.truncate_tgt:
                        tgt = tgt[:config.max_tgt_len]
                    else:
                        continue
                sample['tgt'] = tgt

                img_vec = ' '.join(['0']*1000)
                if imgid != 'none':
                    img_vec = self.img_vecs[imgid+'.jpg']
                sample['img_vec'] = img_vec

                sample['cate'] = cate
                samples.append(sample)

        return samples

    def get_sample(self, text):
        """Build a single sample.

        Args:
            text (str): source + '\t' + target + '\t' + cate + '\t' + imgid
            The string represents a raw sample.

        Returns:
            dict: A sample.
        """
        sample = {}
        src, tgt, cate, imgid = text.split("\t")
        
        src = self.tokenizer(src)
        if config.max_src_len and len(src) > config.max_src_len:
            if config.truncate_src:
                src = src[:config.max_src_len]
        sample['src'] = src

        tgt = self.tokenizer(tgt)
        if config.max_tgt_len and len(tgt) > config.max_tgt_len:
            if config.truncate_tgt:
                tgt = tgt[:config.max_tgt_len]
        sample['tgt'] = tgt

        img_vec = ' '.join(['0'] * config.img_vec_dim)
        if img_vec != '':
            try:
                img_vec = self.img_vecs[imgid]
            except Exception as e:
                img_vec = self.img_vecs[imgid.strip()+'.jpg']
        sample['img_vec'] = img_vec

        sample['cate'] = cate

        x, oov = source2ids(src, self.vocab)
        y = abstract2ids(sample['tgt'], self.vocab, oov)

        output = {
            'source': sample['src'],
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + y + [self.vocab.EOS],
            'x_len': len(sample['src'])+2,
            'y_len': len(sample['tgt'])+2,
            'img_vec': sample['img_vec']
        }
        
        return output

    def build_vocab(self, embed_file: str = None) -> Vocab:
        """Build the vocabulary for the data set.

        Args:
            embed_file (str, optional):
            The file path of the pre-trained embedding word vector.
            Defaults to None.

        Returns:
            vocab.Vocab: The vocab object.
        """
        # word frequency
        word_counts = Counter()
        count_words(word_counts,
                    [sample['src'] + sample['tgt'] for sample in self.samples])
        vocab = Vocab()
        # Filter the vocabulary by keeping only the top k tokens in terms of
        # word frequncy in the data set, where k is the maximum vocab size set
        # in "config.py".
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])
        if embed_file is not None:
            count = vocab.load_embeddings(embed_file)
            logging.info("%d pre-trained embeddings loaded." % count)

        return vocab

    def get_img_vecs(self, img_vecs_path):
        """Get the mapping between image ids and corresponding image vectors read from the image-vector file.

        Args:
            img_vecs_path (str): The path of the image vector file.

        Returns:
            img_vecs (dict): The mapping between image ids and corresponding image vectors
        """        
        img_vecs = {}
        with open(img_vecs_path, 'r') as file:
            for line in file:
                imgid, img_vec = line.strip().split('\t')
                img_vec = list(map(lambda x: float(x), img_vec.split(' ')))
                img_vecs[imgid] = img_vec
        return img_vecs

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if type(item) is int:
            x, oov = source2ids(self.samples[item]['src'], self.vocab)
            y = abstract2ids(self.samples[item]['tgt'], self.vocab, oov)
            return {
                'source': self.samples[item]['src'],
                'x': [self.vocab.SOS] + x + [self.vocab.EOS],
                'OOV': oov,
                'len_OOV': len(oov),
                'y': [self.vocab.SOS] + y + [self.vocab.EOS],
                'x_len': len(self.samples[item]['src'])+2,
                'y_len': len(self.samples[item]['tgt'])+2,
                'img_vec': self.samples[item]['img_vec']
            }
        return [sample[item] for sample in self.samples]


def collate_fn(batch):
    """Split data set into batches and do padding for each batch.

    Returns:
        x_padded (Tensor): Padded source sequences.
        y_padded (Tensor): Padded reference sequences.
        x_len (int): Sequence length of the sources.
        y_len (int): Sequence length of the references.
        OOV (dict): Out-of-vocabulary tokens.
        len_OOV (int): Number of OOV tokens.
        img_vec （list): The image vector.
    """
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item))
                      for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])

    img_vec = torch.tensor(data_batch['img_vec'])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV, img_vec

