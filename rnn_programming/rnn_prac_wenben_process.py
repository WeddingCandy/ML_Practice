# -*- coding: utf-8 -*-
"""
@CREATETIME: 13/07/2018 08:26 
@AUTHOR: Chans
@VERSION: 
"""

import pickle
import pandas as pd
import jieba
import numpy as np
from collections import Counter
import math


def load_data(file_in_path, pickle_text=True, pickle_out_path=None):
    '''
    file_in_path=".../input/training.csv"
    pickle_out_path=".../output/texting.txt"
    '''
    train_data = pd.read_csv(file_in_path, header=None, names=['ind', 'text'])
    texts = train_data['text']
    ind = train_data['ind']
    ind = np.asarray(ind)
    text1 = []
    for text in texts:
        text1.append(" ".join(jieba.cut(text)))
    text1 = [s.split(" ") for s in text1]
    if pickle_text:
        if pickle_out_path is not None:
            dictionary = {'ind':ind, 'texts': text1}
            with open(pickle_out_path, "wb") as f:
                pickle.dump(dictionary, f)
        else:
            print("you should provide pickle_out_path")

    return ind, text1

def _tf(word, count):
    return count[word] / sum(count.values())

def _containing(word, countlist):
    return sum(1 for count in countlist if word in count)

def _idf(word,countlist):
    return math.log(len(countlist)/(1+_containing(word, countlist)))


def add_dict(texts,row_key_word=5, limits=3000):
    countlist = []
    dictionary = set()
    word2index = dict()
    for text in texts:
        countlist.append(Counter(text))
    for count in countlist:
        tfidf = dict()
        for word in count:
            tfidf[word] = _tf(word, count) * _idf(word, countlist)
        sorted_word = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:row_key_word]
        word = [w[0] for w in sorted_word]
        for w in word:
            dictionary.add(w)
        if len(dictionary) > limits+1:
            break
    for i, word in enumerate(dictionary):
        word2index[word] = i+1 #need add the unknown word, index 0
    word2index['UNK'] = 0
    return word2index


def convert_text(texts,row_key_word=5, limits=20000, ispickle=False, pickle_out_path=None):
    textlist = []
    word2index = add_dict(texts, row_key_word, limits)
    for text in texts:
        wordlist = []
        for word in text:
            if word in word2index:
                wordlist.append(word2index[word])
            else:
                wordlist.append(word2index["UNK"])
        textlist.append(wordlist)
    if ispickle is not None:
        with open(pickle_out_path, 'wb') as f:
            pickle.dump(textlist, f)
    return textlist





testfile_in_path="/Users/Apple/datadata/wenben/rnn_test1/testing.csv"
trainfile_in_path="/Users/Apple/datadata/wenben/rnn_test1/training.csv"
test_pickle_out_path="/Users/Apple/datadata/wenben/rnn_test1/regular_text/testing.txt"
train_pickle_out_path="/Users/Apple/datadata/wenben/rnn_test1/regular_text/training.txt"

train_data = load_data(trainfile_in_path,pickle_text=True,pickle_out_path=train_pickle_out_path)
test_data = load_data(testfile_in_path,pickle_text=True,pickle_out_path=test_pickle_out_path)

# with open("/Users/Apple/datadata/wenben/rnn_test1/regular_text/training.txt", 'rb') as f:
#     train_data = pickle.load(f)
# with open("/Users/Apple/datadata/wenben/rnn_test1/regular_text/testing.txt", 'rb') as f:
#     test_data = pickle.load(f)


ind, train_texts = train_data[0], train_data[1]
ind -= 1
_, test_texts = test_data[0], test_data[1]

textlist_train = convert_text(train_texts, row_key_word=7, limits=10000,
                        ispickle=True, pickle_out_path="/Users/Apple/datadata/wenben/rnn_test1/regular_text/textlist_train.txt")
textlist_test = convert_text(test_texts, row_key_word=7, limits=10000,
                        ispickle=True, pickle_out_path="/Users/Apple/datadata/wenben/rnn_test1/regular_text/textlist_test.txt")