#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

################################################################################
#
# Copyright (c) 2017. All Rights Reserved
#
################################################################################
"""
该模块对quora 重复文档进行识别，采用lstm+attention模型；

Authors: Fan Tao (fantao@mail.ustc.edu.cn)
Date:    2017/04/04 11:34:00
"""

# from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
import numpy as np
import os

import data_process
import word2vec_pretrain

EMBED_DIM = 64
HIDDEN_DIM = 100
BATCH_SIZE = 32
NBR_EPOCHS = 1

MODEL_DIR="../data/"

def model(in_file):
    """ 判断文档是否重复的lstm+attention模型；
    """
    # 数据预处理阶段；
    print("data process...")
    ques_pairs = data_process.parse_quora_dul_data(in_file)
    word2idx = data_process.build_vocab(ques_pairs)
    vocab_size = len(word2idx) + 1
    seq_maxlen = data_process.get_seq_maxlen(ques_pairs)
    x_ques1, x_ques2, y, pids = data_process.vectorize_ques_pair(ques_pairs, word2idx, seq_maxlen)
    
    x_ques1train, x_ques1test, x_ques2train, x_ques2test, ytrain, ytest, pidstrain, pidstest = \
    train_test_split(x_ques1, x_ques2, y, pids, test_size=0.2, random_state=42)
    print(x_ques1train.shape, x_ques1test.shape, x_ques2train.shape, x_ques2test.shape, 
      ytrain.shape, ytest.shape, pidstrain.shape, pidstest.shape)
    
    # 计算embeding 初始weight；
    w2v_embedding_model = word2vec_pretrain.train_word2vec(ques_pairs,
                                                           num_features=EMBED_DIM, 
                                                           min_word_count=1, 
                                                           context=5)
    embedding_weights = np.zeros((vocab_size, EMBED_DIM))
    for word, index in word2idx.iteritems():
        if word in w2v_embedding_model:
            embedding_weights[index, :] = w2v_embedding_model[word]
        else:
            embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, 
                                                            w2v_embedding_model.vector_size)
    
    # 建立模型；
    print("Building model...")
    ques1_enc = Sequential()
    ques1_enc.add(Embedding(output_dim=EMBED_DIM, 
                            input_dim=vocab_size,
                            input_length=seq_maxlen,
                       weights=[embedding_weights]))
    ques1_enc.add(LSTM(HIDDEN_DIM,  return_sequences=True))
    ques1_enc.add(Dropout(0.3))
    
    ques2_enc = Sequential()
    ques2_enc.add(Embedding(output_dim=EMBED_DIM, 
                            input_dim=vocab_size,
                            input_length=seq_maxlen,
                       weights=[embedding_weights]))
    ques2_enc.add(LSTM(HIDDEN_DIM, return_sequences=True))
    ques2_enc.add(Dropout(0.3))
    
    # attention model
    attn = Sequential()
    attn.add(Merge([ques1_enc, ques2_enc], mode="dot", dot_axes=[1, 1]))
    attn.add(Flatten())
    attn.add(Dense((seq_maxlen * HIDDEN_DIM)))
    attn.add(Reshape((seq_maxlen, HIDDEN_DIM)))
    
    model = Sequential()
    model.add(Merge([ques1_enc, attn], mode="sum"))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    print("Training...")
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "quora_dul_best_lstm_atten.hdf5"),
        verbose=1, save_best_only=True)
    model.fit([x_ques1train, x_ques2train], ytrain, batch_size=BATCH_SIZE,
              epochs=NBR_EPOCHS, validation_split=0.1,
              verbose=2,
              callbacks=[checkpoint])
    
    # predict
    print ("predict...")
    y_test_pred = model.predict_classes([x_ques1test, x_ques2test], batch_size=BATCH_SIZE)
    data_process.pred_save("../data/y_test.pred", y_test_pred, ytest, pidstest)
    
    print("Evaluation...")
    loss, acc = model.evaluate([x_ques1test, x_ques2test], ytest, batch_size=BATCH_SIZE)
    print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))
    
    model.save_weights(os.path.join(MODEL_DIR, "quora_dul_lstm_atten-final.hdf5"))
    with open(os.path.join(MODEL_DIR, "quora_dul_lstm.json"), "wb") as fjson:
        fjson.write(model.to_json())
    
    model.load_weights(filepath=os.path.join(MODEL_DIR, "quora_dul_best_lstm_atten.hdf5"))
    loss, acc = model.evaluate([x_ques1test, x_ques2test], ytest, batch_size=BATCH_SIZE)
    print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
    

if __name__ == '__main__':
    model("../data/quora_duplicate_questions.tsv")
    
    
    
    