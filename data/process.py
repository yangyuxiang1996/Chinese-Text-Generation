#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-07-13 20:16:37
LastEditTime: 2021-01-26 00:22:29
FilePath: /Assignment2-3/data/process.py
Desciption: Process a raw dataset into a sample file and partition it into train, dev, test sets.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''


import sys
import os

import json

sys.path.append('..')
from data_utils import write_samples, partition
import config

samples = set()
# Read json file.

json_files = [os.path.join(config.data_folder, path) for path in os.listdir(config.data_folder) if '.json' in path]

for json_file in json_files:
    with open(json_file, 'r', encoding='utf8') as file:
        jsf = json.load(file)

    for key, val in jsf.items():
        imgid = key  # Get image id.
        source = val['src'].replace('。', '').replace('，', '')  # Get source.
        cate = val['cate'] # Get category.
        targets = val['tgt'] # Get targets.

        # Create a sample for every target(reference).
        for target in targets:
            sample = source + '\t' + target + '\t' + cate + '\t' + imgid
            samples.add(sample)

write_path = config.sample_path

print('write_path: ', write_path)
write_samples(samples, write_path)
partition(samples)
