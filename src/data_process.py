#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
该模块对quora 重复文档进行数据处理；

Authors: Fan Tao (fantao@mail.ustc.edu.cn)
Date:    2017/04/04 11:34:00
"""

import numpy as np
import codecs
import re
from keras.preprocessing.sequence import pad_sequences
import collections


def tokenize(sent):
    """切句子；
    """
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_quora_dul_data(in_file):
    """解析quara 重复文档文件；
    """
    ques_pairs = []
    with codecs.open(in_file, 'rb') as fi:
        fi.readline()        
        for line in fi:
            if len(line) == 0:
                continue
            splits = line.split("\t")
            if len(splits) < 6:
                continue
            pid = splits[0]
            ques1 = splits[3]  
            ques2 = splits[4] 
            is_dul = int(splits[5])  
            
            ques1_token = tokenize(ques1) 
            ques2_token = tokenize(ques2) 
            
            ques_pairs.append((ques1_token, ques2_token, is_dul, pid))
             
    return ques_pairs


def build_vocab(ques_pairs):
    """ 建立token ->id 索引；
    """
    wordcounts = collections.Counter()
    
    for pair in ques_pairs:
        for w in pair[0]:
            wordcounts[w] += 1
        for w in pair[1]:
            wordcounts[w] += 1
            
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2idx = {w: i + 1 for i, w in enumerate(words)}  # 0 = mask
    
    return word2idx


def get_seq_maxlen(ques_pairs):
    """ 计算最大句子长度；    
    """
    max_ques1_len = max([len(pair[0]) for pair in ques_pairs])
    max_ques2_len = max([len(pair[1]) for pair in ques_pairs])
    max_seq_len = max([max_ques1_len, max_ques2_len])
    
    return max_seq_len

    
def vectorize_ques_pair(ques_pairs, word2idx, seq_maxlen): 
    """ 对question pair 进行id向量化；
    """
    x_ques1 = []
    x_ques2 = []
    y = []
    pids = []
    for pair in ques_pairs:
        x_ques1.append([word2idx[w] for w in pair[0]])
        x_ques2.append([word2idx[w] for w in pair[1]])
        y.append((np.array([0, 1]) if pair[2] == 1 else np.array([1, 0])))
        pids.append(pair[3])
                 
    x_ques1 = pad_sequences(x_ques1, maxlen=seq_maxlen)
    x_ques2 = pad_sequences(x_ques2, maxlen=seq_maxlen)
    y = np.array(y)
    pids = np.array(pids)
    
    return x_ques1, x_ques2, y, pids


def pred_save(out_file, y_preds, y_trues, ids):
    """ 输出预测结果；
    """
    with open(out_file, "w") as fo:
        for i in range(len(y_preds)):
            pred = y_preds[i]
            truelabel = y_trues[i][1]
            pid = ids[i]
            fo.write("%s,%s,%s\n" % (str(pid), str(pred), str(truelabel)))
            