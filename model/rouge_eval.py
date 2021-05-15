#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-07-26 16:13:08
LastEditTime: 2021-01-26 00:41:30
FilePath: /Assignment2-3/model/rouge_eval.py
Desciption: Evaluate the model with ROUGE score.
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import sys
import os

from rouge import Rouge

sys.path.append('..')

from predict import Predict
from utils import timer
import config
from dataset import SamplesDataset


class RougeEval():
    def __init__(self, path, vocab=None):
        self.predict = Predict()
        self.rouge = Rouge()
        self.testset = SamplesDataset(path, vocab=self.predict.vocab)
        print('test vocab: ', len(self.testset.vocab))
        self.refs = self.process()
        self.hypos = self.build_hypos()
        self.scores = self.get_scores()

    def process(self):
        tgts = self.testset['tgt']
        refs = []
        for tgt in tgts:
            ref = ' '.join(tgt).replace('。', '.')
            refs.append(ref)
        print(f'Test set contains {len(self.testset)} samples.')
        return refs

    @timer('building hypotheses')
    def build_hypos(self):
        """Generate hypos for the dataset.

        Args:
            predict (predict.Predict()): The predictor instance.
        """
        print('Building hypotheses.')
        count = 0
        exception_count = 0
        hypos = []
        for sample in self.testset:
#             print(' '.join(sample['source']))
            count += 1
            if count % 100 == 0:
                print(count)
#             try:
            hypo = self.predict.predict(sample['x'], sample['OOV'], sample['img_vec'])
            hypos.append(hypo)
#             except Exception as e:
#                 print(sample)
#                 print(e)
#                 exit(0)
#             try:
#                 self.hypos.append(predict.predict(sample['x'], sample['OOV'], sample['img_vec']))
#             except:
#                 exception_count += 1
#                 print('exception: ', exception_count)
#                 self.hypos.append('')
        return hypos

    def get_scores(self):
        assert len(self.hypos) > 0, 'Build hypotheses first!'
        print('Calculating average rouge scores.')
        for hypo, ref in zip(self.hypos, self.refs):
            print(hypo, ' ||| ', ref)
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)

    
    
    def one_sample(self, hypo, ref):
        return self.rouge.get_scores(hypo, ref)[0]
    
    
    def get_metric(self, metric):
        return self.scores[metric]['p']


rouge_eval = RougeEval(config.test_data_path)

result = rouge_eval.scores
print('rouge1: ', result['rouge-1'])
print('rouge2: ', result['rouge-2'])
print('rougeL: ', result['rouge-l'])
with open(config.rouge_path, 'a') as file:
    file.write(config.model_name)
    for r, metrics in result.items():
        file.write(r+'\n')
        for metric, value in metrics.items():
            file.write(metric+': '+str(value*100))
            file.write('\n')
