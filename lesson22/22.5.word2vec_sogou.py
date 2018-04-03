# encoding: utf-8

from time import time
from gensim.models import Word2Vec
import gensim.models.word2vec
import os
import re
import codecs


def deal_special(s):
    pattern = re.compile(r"[^\u4e00-\u9f5aa-zA-Z0-9]")
    return pattern.sub('', s)

class LoadCorpora(object):
    def __init__(self, dir_name):
        self.path = dir_name

    def __iter__(self):#迭代器
        count = 0
        for file_name in os.listdir(self.path):
            if file_name != '.DS_Store':
                path_name = os.path.join(self.path, file_name)
                print(path_name)
                f = open(path_name, 'r', encoding='utf-8')  # ,encoding='gb18030','ignore'
                # ff = open(corpora_path_2,'r' ,encoding='utf16')
                # for ll in ff :print(ll)
                for line in f:
                    # line = re.sub("[^\u4e00-\u9f5aa-zA-Z0-9]",' ',line) #别随便消除空格呀！！！
                    # try:
                        yield [word.strip() for word in line.split(' ')]
                    # except Exception as e:
                    #     print("Exception: {}".format(e))
            else:
                print('A wrong.')



            #

def print_list(a):
    for i, s in enumerate(a):
        if i != 0:

            print('+', end=' ')
        print(s, end=' ')





if __name__ == '__main__':
    corpora_path = '/Volumes/d/data/jieba_cut/'
    # corpora_path = '/Volumes/d/data/jieba_cut_test/'
    corpora_model_path = '/Users/Apple/PycharmProjects/learn_ml/lesson22/corpora_data'
    model_name = '/Users/Apple/PycharmProjects/learn_ml/lesson22/corpora_data/200806.model'
    if not os.path.exists(model_name):
        print('step 1 is runnning')
        sentences = LoadCorpora(corpora_path)
        print(sentences)
        t_start = time()
        model = Word2Vec(sentences, size=200, min_count=1, workers=8)  # 词向量维度为200，丢弃出现次数少于5次的词
        print('step 1 is done.')
        model.save(model_name)
        print('OK:', time() - t_start)
    model = Word2Vec.load(model_name)
    print('model.wv.vocab = ', type(model.wv.vocab), len(model.wv.vocab))
    for i, word in enumerate(model.wv.vocab):
        # print(word,i, end=' ')
        if i % 50 == 49:
            print()
    # print()

    intrested_words = ('中国','手机', '学习', '名义') #'中国',
    print('特征向量：')
    for word in intrested_words:
        print(word, len(model[word]), model[word])
    for word in intrested_words:
        result = model.most_similar(word)
        print('与', word, '最相近的词：')
        for w, s in result:
            print('\t', w, s)

    words = ('中国', '祖国', '毛泽东')
    for i in range(len(words)):
        w1 = words[i]
        for j in range(i+1, len(words)):
            w2 = words[j]
            print('%s 和 %s 的相似度为：%.6f' % (w1, w2, model.similarity(w1, w2)))

    print('========================') #这一步做相近词和反义词的设定
    opposites = ((['中国', '城市'], ['学生']),
                 (['男', '工作'], ['女']),
                 (['俄罗斯', '美国', '英国'], ['日本'])) #写成元组的方式
    for positive, negative in opposites:
        result = model.most_similar(positive=positive, negative=negative)
        print_list(positive)
        print('-', end=' ')
        print_list(negative)
        print('：')
        for word, similar in result:
            print('\t', word, similar)

    print('========================') #这一步查找离群词的情况doesnt_match()
    words = '苹果 三星 小米 联想 华为 海尔 格力'
    print(words, '离群词：', model.doesnt_match(words.split(' ')))
    words = '苹果 三星 美的 海尔'
    print(words, '离群词：', model.doesnt_match(words.split(' ')))
    words = '中国 日本 韩国 美国 北京'
    print(words, '离群词：', model.doesnt_match(words.split(' ')))
