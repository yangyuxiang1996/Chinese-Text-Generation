#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-04-10 08:47:00
LastEditors: yangyuxiang
LastEditTime: 2021-04-10 08:47:14
FilePath: /Assignment2-2/model/w2v.py
'''
from typing import Callable
from gensim.models import word2vec
import logging
import config
from utils import simple_tokenizer
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S'
                    )

class PairDataset(object):
    """The class represents source-reference pairs.

    """
    def __init__(self,
                 filename,
                 tokenize: Callable = simple_tokenizer,
                 max_src_len: int = None,
                 max_tgt_len: int = None,
                 truncate_src: bool = False,
                 truncate_tgt: bool = False):
        logging.info("Reading dataset %s..." % filename)
        self.filename = filename
        self.pairs = []

        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                # Split the source and reference by the <sep> tag.
                pair = line.strip().split('<sep>')
                if len(pair) != 2:
                    print("Line %d of %s is malformed." % (i, filename))
                    print(line)
                    continue
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                self.pairs.append((src, tgt))
        logging.info("%d pairs." % len(self.pairs))


if __name__ == "__main__":
    dataset = PairDataset(config.data_path,
                            max_src_len=config.max_src_len,
                            max_tgt_len=config.max_tgt_len,
                            truncate_src=config.truncate_src,
                            truncate_tgt=config.truncate_tgt)
                            
    val_dataset = PairDataset(config.val_data_path,
                                max_src_len=config.max_src_len,
                                max_tgt_len=config.max_tgt_len,
                                truncate_src=config.truncate_src,
                                truncate_tgt=config.truncate_tgt)

    test_dataset = PairDataset(config.val_data_path,
                                max_src_len=config.max_src_len,
                                max_tgt_len=config.max_tgt_len,
                                truncate_src=config.truncate_src,
                                truncate_tgt=config.truncate_tgt)

    pairs = dataset.pairs
    pairs.extend(val_dataset.pairs)
    pairs.extend(test_dataset.pairs)
    
    corpus = []
    for pair in pairs:
        src, tgt = pair
        corpus.append(src+tgt)
    logging.info(corpus[0])

    model = word2vec.Word2Vec(size=300, min_count=1, workers=4)
    model.build_vocab(corpus)
    model.train(corpus, epochs=10, total_examples=model.corpus_count)

    model.wv.save_word2vec_format('../files/embedding.txt', binary=False)



