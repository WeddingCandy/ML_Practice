# -*- coding: utf-8 -*-
#!/usr/bin/python
import sys
import numpy as np
import gensim
import gensim.models.doc2vec as d2v
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import re
from gensim.utils import simple_preprocess

##读取文档下的所有数据，变成一个文本
def get_namelist(path):
    # namelist={}
    nameList=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            # pattern = re.compile('\.txt') #if needed classifications in the end ,use this .
            # classfication = pattern.sub('',os.path.join(file))
            filename = os.path.join(root+'/'+file)
            nameList.append(filename)
            # namelist[filename] = classfication
    return nameList


##合成一个文档
def merge_to_list(pathlist):
    contentslist =[]
    count = 0
    for path in pathlist:
        try:
            with open(path,'r',encoding='utf-8') as f :
                count += 1
                # print(content)
                content = (f.readlines()[0]).replace('<br />',' ').replace('\\', ' ').strip()
                content = simple_preprocess(content)
                contentslist.append(content)

        except Exception as e :
            print(e)
            continue
    return contentslist



##读取并预处理数据
def get_dataset(pos_file,neg_file,unsup_file):
    LabeledSentence = d2v.LabeledSentence
    pos_reviews = merge_to_list(get_namelist(pos_file))
    neg_reviews = merge_to_list(get_namelist(neg_file))
    unsup_reviews = merge_to_list(get_namelist(unsup_file))

    #使用1表示正面情感，0为负面
    x = np.concatenate((pos_reviews, neg_reviews))
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(pos_reviews))))
    #将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(x_train.shape,y_train.shape)

    #对英文做简单的数据清洗预处理，中文根据需要进行修改
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z[0].lower().replace('\n','') for z in corpus]
        corpus = [z[0].replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z[0].replace(c, ' %s '%c) for z in corpus]
        corpus = [z[0].split() for z in corpus]
        return corpus
    #
    # x_train = cleanText(x_train)
    # x_test = cleanText(x_test)
    # unsup_reviews = cleanText(unsup_reviews)

    #Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    #我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    return x_train,x_test,unsup_reviews,y_train, y_test

##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

##对数据进行训练
def train(x_train,x_test,unsup_reviews,size ,epoch_num):
    #实例DM和DBOW模型
    # model_dm = d2v.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = d2v.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    #使用所有的数据建立词典
    # model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

    xxx = []
    xxx.extend(x_train)
    xxx.extend(x_test)
    xxx.extend(unsup_reviews)


    # model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
    model_dbow.build_vocab(xxx)

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    yyy =[]
    yyy.extend(x_train)
    yyy.extend(unsup_reviews)
    all_train_reviews = yyy
    for epoch in range(epoch_num):
        perm = np.random.permutation(all_train_reviews)
        # model_dm.train(all_train_reviews[perm])
        model_dbow.train(all_train_reviews)

    #训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        # model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])

    return model_dbow #,model_dm

##使用分类器对文本向量进行分类训练
def Classifier(train_vecs,y_train,test_vecs, y_test):
    #使用sklearn的SGD分类器
    lr = SGDClassifier(loss='log', penalty='l2')
    lr.fit(train_vecs, y_train)

    print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))

    return lr

##绘出ROC曲线，并计算AUC
def ROC_curve(lr,test_vecs,y_test):
    pred_probas = lr.predict_proba(test_vecs)[:,1]
    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()




##运行模块
if __name__ == "__main__":


    pos_file = '/Volumes/d/data/aclImdb/train_test/pos'
    neg_file = '/Volumes/d/data/aclImdb/train_test/neg'
    unsup_file = '/Volumes/d/data/aclImdb/train_test/unsup'
    #设置向量维度和训练次数
    size,epoch_num = 400,10
    #获取训练与测试数据及其类别标注
    x_train,x_test,unsup_reviews,y_train, y_test = get_dataset(pos_file,neg_file,unsup_file)
    #对数据进行训练，获得模型
    model_dm,model_dbow = train(x_train,x_test,unsup_reviews,size,epoch_num)
    #从模型中抽取文档相应的向量
    train_vecs,test_vecs = getVecs(model_dm,model_dbow,size)
    #使用文章所转换的向量进行情感正负分类训练
    lr=Classifier(train_vecs,y_train,test_vecs, y_test)
    #画出ROC曲线
    ROC_curve(lr,y_test)